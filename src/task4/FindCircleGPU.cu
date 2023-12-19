#include "FindCircle.cuh"

#define KERNEL_SIZE 3
#define IMAGE_CHANNELS 3
#define BLOCK_SIZE 16

#define SEED 30

#define BLUR_SIZE 17
#define CIRCLE_THICKNESS 3
#define N 5
#define K 100
#define THRESHOLD_RADIUS 7 // 5 for inner circle
#define THRESHOLD_COUNT 600 //100 for inner circle

using namespace fourthTask;


struct Point
{
	int x, y;
	__device__ Point() {}
	__device__ Point(int _x, int _y) : x(_x), y(_y) {}

	__device__ static double dist(const Point& first, const Point& second)
	{
		double _x = first.x - second.x;
		double _y = first.y - second.y;
		return std::sqrt(_x * _x + _y * _y);
	}

	__device__ Point& operator=(const Point& in)
	{
		x = in.x;
		y = in.y;
		return *this;
	}
};


struct Matrix
{
	int height, width;
	double** elements;

	__device__ Matrix(int _height, int _width) : height(_height), width(_width)
	{
		elements = new double* [height];
		for (int i = 0; i < height; i++)
		{
			elements[i] = new double[width];
		}
	}

	template <int _height, int _width>
	__device__ Matrix(double (&_elements)[_height][_width]) : height(_height), width(_width)
	{
		elements = new double* [height];
		for (int i = 0; i < height; i++)
		{
			elements[i] = new double[width];
			for (int j = 0; j < width; j++)
			{
				elements[i][j] = _elements[i][j];
			}
		}
	}

	__device__ static double sum(const Matrix& in)
	{
		double sum = 0;
		for (int i = 0; i < in.height; ++i)
		{
			for (int j = 0; j < in.width; ++j)
			{
				sum += in.elements[i][j];
			}
		}

		return sum;
	}

	__device__ static void subtract(Matrix& in, const double element)
	{
		for (int i = 0; i < in.height; ++i)
		{
			for (int j = 0; j < in.width; ++j)
			{
				in.elements[i][j] -= element;
			}
		}
	}

	__device__ static  double mulSum(const Matrix& first, const Matrix& second)
	{
		Matrix result(first.height, first.width);

		for (int i = 0; i < first.height; i++)
		{
			for (int j = 0; j < first.width; j++)
			{
				result.elements[i][j] = first.elements[i][j] * second.elements[i][j];
			}
		}

		return sum(result);
	}

	__device__ static double mulSum(const Matrix& first, const Matrix& second, const Matrix& third)
	{
		Matrix result(first.height, first.width);

		for (int i = 0; i < first.height; i++)
		{
			for (int j = 0; j < first.width; j++)
			{
				result.elements[i][j] = first.elements[i][j] * second.elements[i][j] * third.elements[i][j];
			}
		}

		return sum(result);
	}

	__device__ static void multiply(const Matrix& first, const Matrix& second, Matrix& result)
	{
		for (int i = 0; i < first.height; i++)
		{
			for (int j = 0; j < second.width; j++)
			{
				result.elements[i][j] = 0;

				for (int k = 0; k < second.height; k++)
				{
					result.elements[i][j] += first.elements[i][k] * second.elements[k][j];
				}
			}
		}
	}

	__device__ Matrix inverse2D()
	{
		Matrix temp(2, 2);

		double det = elements[0][0] * elements[1][1] - elements[0][1] * elements[1][0];

		temp.elements[0][0] = elements[1][1] / det;
		temp.elements[0][1] = -elements[0][1] / det;
		temp.elements[1][0] = -elements[1][0] / det;
		temp.elements[1][1] = elements[0][0] / det;

		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				elements[i][j] = temp.elements[i][j];
			}
		}

		return *this;
	}
};


__host__ unsigned int* GetCircleData(const cv::Mat& inputImage, int& numOfEdges)
{
	cv::Mat grayImage;
	cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

	// Blur the image for better edge detection
	cv::Mat bluredImage;
	cv::GaussianBlur(grayImage, bluredImage, cv::Size(BLUR_SIZE, BLUR_SIZE), 0);

	// https://en.wikipedia.org/wiki/Canny_edge_detector
	cv::Mat edges;
	cv::Canny(bluredImage, edges, 100, 200);

	// Save filtered image
	cv::imwrite("./src/task4/FilteredImage.png", edges);

	// Create array of edge Points
	unsigned int* edgesPoints = new unsigned int[2 * edges.rows * edges.cols];

	for (int row = 0; row < edges.rows; ++row)
	{
		for (int col = 0; col < edges.cols; ++col)
		{
			if (edges.data[row * edges.cols + col] == 255)
			{
				edgesPoints[numOfEdges] = col;
				edgesPoints[numOfEdges + 1] = row;
				numOfEdges += 2;
			}
		}
	}

	return edgesPoints;
}


__device__ Point* GetAllPoints(const unsigned int* allEdges, const int numOfAllPoints)
{
	Point* allPoints = new Point[numOfAllPoints];
	int j = 0;

	for (int i = 0; i < numOfAllPoints; i++)
	{
		
		allPoints[i] = Point(allEdges[j], allEdges[j + 1]);
		j += 2;
	}

	return allPoints;
}


__device__ void SwapRandomPoints(Point* allPoints, const int index1, const int index2)
{
	Point temp = allPoints[index1];
	allPoints[index1] = allPoints[index2];
	allPoints[index2] = temp;
}


__device__ Point* GetRandomPoints(Point* allPoints, const int numOfAllPoints)
{
	curandState s;
	curand_init(SEED, 0, 0, &s);

	for (int i = 0; i < numOfAllPoints; i++)
	{
		int randomIndex = numOfAllPoints * curand_uniform(&s) - 1;
		SwapRandomPoints(allPoints, i, randomIndex);
		randomIndex = numOfAllPoints * curand_uniform(&s) - 1;
		SwapRandomPoints(allPoints, numOfAllPoints - i, randomIndex);
	}

	Point* chosenPoints = new Point[N];

	for (int i = 0; i < N; i++)
	{
		chosenPoints[i] = allPoints[i];
	}

	return chosenPoints;
}


__device__ void GetCircleParametersLeastSquares(const Point* points, const int numOfPoints, Point& circleCenter, double& circleRadius)
{
	// https://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf
	Matrix ui(1, numOfPoints), vi(1, numOfPoints), temp(1, numOfPoints);

	for (int i = 0; i < numOfPoints; ++i)
	{
		ui.elements[0][i] = (double) points[i].x;
		vi.elements[0][i] = (double) points[i].y;
	}

	double averageX = Matrix::sum(ui) / (double)numOfPoints;
	double averageY = Matrix::sum(vi) / (double)numOfPoints;

	Matrix::subtract(ui, averageX);
	Matrix::subtract(vi, averageY);

	double Suu = Matrix::mulSum(ui, ui);
	double Svv = Matrix::mulSum(vi, vi);
	double Suv = Matrix::mulSum(ui, vi);

	double Suuu = Matrix::mulSum(ui, ui, ui);
	double Svvv = Matrix::mulSum(vi, vi, vi);
	double Suvv = Matrix::mulSum(ui, vi, vi);
	double Svuu = Matrix::mulSum(vi, ui, ui);

	double rightSideArray[2][1] = { 0.5 * (Suuu + Suvv), 0.5 * (Svvv + Svuu) };
	double leftSideArray[2][2] = { {Suu, Suv}, {Suv, Svv} };

	Matrix rightSideVector(rightSideArray);
	Matrix leftSideMatrix(leftSideArray);

	Matrix centerVector(2, 1);
	Matrix::multiply(leftSideMatrix.inverse2D(), rightSideVector, centerVector);
	
	double dataX = centerVector.elements[0][0], dataY = centerVector.elements[1][0];

	circleCenter = Point((int)(dataX + averageX), (int)(dataY + averageY));
	circleRadius = (int)std::sqrt(dataX * dataX + dataY * dataY + (Suu + Svv) / numOfPoints);
}


__device__ Point* GetAllInlinePoints(const Point* allPoints, const int numOfAllPoints, const Point circleCenter, const double circleRadius, int& numOfInlinePoints)
{
	Point* inlinePoints = new Point[numOfAllPoints];

	for (int i = 0; i < numOfAllPoints; ++i)
	{
		double pointRadius = Point::dist(allPoints[i], circleCenter);

		if ((pointRadius > (circleRadius - THRESHOLD_RADIUS)) && (pointRadius < (circleRadius + THRESHOLD_RADIUS)))
		{
			inlinePoints[numOfInlinePoints] = allPoints[i];
			++numOfInlinePoints;
		}
	}

	if (numOfInlinePoints != 0)
	{
		return inlinePoints;
	}

	return new Point();
}


__device__ double CalculateMeanError(const Point* inlinePoints, const int numOfInlinePoints, const Point circleCenter, const double circleRadius)
{
	double error = 0;

	for (int i = 0; i < numOfInlinePoints; ++i)
	{
		double pointRadius = Point::dist(inlinePoints[i], circleCenter);
		error += std::abs(circleRadius - pointRadius);
	}

	return error / numOfInlinePoints;
}


__device__ void GetCandidateCircleParameters(unsigned int* allEdges, const int numOfAllPoints, Point& circleCenter, double& circleRadius, double& error)
{
	// https://sdg002.github.io/ransac-circle/index.html
	Point* allPoints = GetAllPoints(allEdges, numOfAllPoints);
	Point* chosenEdges = GetRandomPoints(allPoints, numOfAllPoints);

	GetCircleParametersLeastSquares(chosenEdges, N, circleCenter, circleRadius);

	int numOfInlinePoints = 0;
	Point* inlinePoints = GetAllInlinePoints(allPoints, numOfAllPoints, circleCenter, circleRadius, numOfInlinePoints);

	if (numOfInlinePoints > THRESHOLD_COUNT)
	{
		GetCircleParametersLeastSquares(inlinePoints, numOfInlinePoints, circleCenter, circleRadius);

		numOfInlinePoints = 0;
		inlinePoints = GetAllInlinePoints(allPoints, numOfAllPoints, circleCenter, circleRadius, numOfInlinePoints);

		if (numOfInlinePoints > THRESHOLD_COUNT)
		{
			error = CalculateMeanError(inlinePoints, numOfInlinePoints, circleCenter, circleRadius);
			return;
		}
	}

	error = -1;
}


__global__ void IterateOverCandidates(unsigned int* edgesPoints, const int numOfAllPoints, Point * circleCenters, double* circleRadiuses, double* errors, int& numOfCandidates)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < K)
	{
		Point circleCenter;
		double circleRadius, error;
		GetCandidateCircleParameters(edgesPoints, numOfAllPoints, circleCenter, circleRadius, error);

		if (error != -1)
		{
			circleCenters[numOfCandidates] = circleCenter;
			circleRadiuses[numOfCandidates] = circleRadius;
			errors[numOfCandidates] = error;
			++numOfCandidates;
		}
	}
}


__host__ std::pair<cv::Point, double> ChooseBestParameters(const Point* circleCenters, const double* circleRadiuses, const double* errors, const int numOfCandidates)
{
	assert(numOfCandidates != 0);

	int bestParameterIndex = 0;
	double tempError = 10000;

	for (int i = 0; i < numOfCandidates; ++i)
	{
		if (errors[i] < tempError)
		{
			tempError = errors[i];
			bestParameterIndex = i;
		}
	}

	std::pair <cv::Point, double> circleParameters(cv::Point(circleCenters[bestParameterIndex].x, circleCenters[bestParameterIndex].y), circleRadiuses[bestParameterIndex]);

	return circleParameters;
}


__host__ std::pair<cv::Point, double> GetOptimalParameters(unsigned int* edgesPoints, const int numOfEdges)
{
	int numOfCandidates = 0;
	Point* circleCenters, * circleCentersCPU = new Point[K];
	double* circleRadiuses, * circleRadiusesCPU = new double[K], * errors, * errorsCPU = new double[K];

	cudaMallocManaged(&circleCenters, K * sizeof(Point));
	cudaMallocManaged(&circleRadiuses, K * sizeof(double));
	cudaMallocManaged(&errors, K * sizeof(double));

	dim3 blockSize(BLOCK_SIZE);
	dim3 numBlocks(ceil(K / (double) BLOCK_SIZE));

	IterateOverCandidates <<<numBlocks, blockSize>>> (edgesPoints, (int)(numOfEdges / 2), circleCenters, circleRadiuses, errors, numOfCandidates);
	cudaDeviceSynchronize();

	cudaMemcpy(errors, errorsCPU, K, cudaMemcpyDeviceToHost);
	cudaMemcpy(circleCenters, circleCentersCPU, K, cudaMemcpyDeviceToHost);
	cudaMemcpy(circleRadiuses, circleRadiusesCPU, K, cudaMemcpyDeviceToHost);

	// Wait for GPU to finish
	cudaDeviceSynchronize();

	return ChooseBestParameters(circleCentersCPU, circleRadiusesCPU, errorsCPU, numOfCandidates);
}


void fourthTask::FindCircleGPU(const std::string pathToImage)
{
	// read data
	cv::Mat inputImage = cv::imread(pathToImage);
	cv::Mat outputImage = inputImage.clone();
	int numOfEdges = 0;

	// select data
	unsigned int* edgesPoints = GetCircleData(inputImage, numOfEdges);
	unsigned int* edgesPointsGPU;

	// allocate data
	{
		cudaMallocManaged(&edgesPointsGPU, numOfEdges * sizeof(unsigned int));
		cudaMemcpy(edgesPoints, edgesPointsGPU, numOfEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
	}

	// get optimal circle parameters
	std::pair<cv::Point, double> circleParameters = GetOptimalParameters(edgesPointsGPU, numOfEdges);

	// add circle to the image
	{
		cv::circle(outputImage, circleParameters.first, circleParameters.second, cv::Scalar(0, 0, 0), CIRCLE_THICKNESS);
		cv::imwrite("./src/task4/OutputImage.png", outputImage);
	}

	// free memorey
	{
		delete[] edgesPoints;
		cudaFree(edgesPointsGPU);
	}
}
