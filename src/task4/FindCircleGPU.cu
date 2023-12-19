#include "FindCircle.cuh"

#define IMAGE_CHANNELS 3
#define BLOCK_SIZE 16

#define BLUR_SIZE 17
#define CIRCLE_THICKNESS 3

#define N 5
#define K 50
#define THRESHOLD_RADIUS 7 // 5 for inner circle
#define THRESHOLD_COUNT 100 //100 for inner circle

#define NUM_OF_ALL_POINTS 2747

using namespace fourthTask;


namespace Point
{
	__device__ static double dist(const int* first, const int* second)
	{
		double _x = first[0] - second[0];
		double _y = first[1] - second[1];
		return std::sqrt(_x * _x + _y * _y);
	}
}


namespace Matrix
{
	__device__ static double sum(const double* in, const int numOfData)
	{
		double sum = 0;
		for (int i = 0; i < numOfData; ++i)
		{
			sum += in[i];
		}

		return sum;
	}

	__device__ static void subtract(double* in, const double element, const int numOfData)
	{
		for (int i = 0; i < numOfData; ++i)
		{
			in[i] -= element;
		}
	}

	__device__ static  double mulSum(const double* first, const double* second, const int numOfData)
	{
		double sum = 0;
		for (int i = 0; i < numOfData; i++)
		{
			sum += first[i] * second[i];
		}

		return sum;
	}

	__device__ static double mulSum(const double* first, const double* second, const double* third, const int numOfData)
	{
		double sum = 0;
		for (int i = 0; i < numOfData; i++)
		{
			sum += first[i] * second[i] * third[i];
		}

		return sum;
	}
}


__host__ int* GetCircleData(const cv::Mat& inputImage, int& numOfEdges)
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
	int* edgesPoints = new int[2 * edges.rows * edges.cols];

	for (int row = 0; row < edges.rows; ++row)
	{
		for (int col = 0; col < edges.cols; ++col)
		{
			if (edges.data[row * edges.cols + col] == 255)
			{
				edgesPoints[2 * numOfEdges] = col;
				edgesPoints[2 * numOfEdges + 1] = row;
				++numOfEdges;
			}
		}
	}

	return edgesPoints;
}


__device__ void GetRandomPoints(const int* allPoints, int* chosenPoints)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	curandState s;
	curand_init(index, 0, 0, &s);

	for (int i = 0; i < N; i++)
	{
		int randomIndex = curand(&s) % NUM_OF_ALL_POINTS;
		//int randomIndex = (NUM_OF_ALL_POINTS - 2) * curand_uniform(&s);
		randomIndex /=  2;
		chosenPoints[2 * i] = allPoints[2 * randomIndex];
		chosenPoints[2 * i + 1] = allPoints[2 * randomIndex + 1];
	}
}


__device__ void GetCircleParametersLeastSquares(const int* points, const int numOfPoints, int* circleCenter, double& circleRadius)
{
	// https://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf
	double* ui = new double[NUM_OF_ALL_POINTS], * vi = new double[NUM_OF_ALL_POINTS];

	for (int i = 0; i < NUM_OF_ALL_POINTS; ++i)
	{
		if (i < numOfPoints)
		{
			ui[i] = (double)points[2 * i];
			vi[i] = (double)points[2 * i + 1];
		}
		else
		{
			ui[i] = 0.0;
			vi[i] = 0.0;
		}
	}

	double averageX = Matrix::sum(ui, numOfPoints) / (double)numOfPoints;
	double averageY = Matrix::sum(vi, numOfPoints) / (double)numOfPoints;

	Matrix::subtract(ui, averageX, numOfPoints);
	Matrix::subtract(vi, averageY, numOfPoints);

	double Suu = Matrix::mulSum(ui, ui, numOfPoints);
	double Svv = Matrix::mulSum(vi, vi, numOfPoints);
	double Suv = Matrix::mulSum(ui, vi, numOfPoints);

	double rightSideX = 0.5 * (Matrix::mulSum(ui, ui, ui, numOfPoints) + Matrix::mulSum(ui, vi, vi, numOfPoints));
	double rightSideY = 0.5 * (Matrix::mulSum(vi, vi, vi, numOfPoints) + Matrix::mulSum(vi, ui, ui, numOfPoints));

	double det = Suu * Svv - Suv * Suv;
	double dataX = (Svv - Suv) * rightSideX / det, dataY = (Suu - Suv) * rightSideY / det;

	circleCenter[0] = dataX + averageX;
	circleCenter[1] = dataY + averageY;
	circleRadius = (int)std::sqrt(dataX * dataX + dataY * dataY + (Suu + Svv) / numOfPoints);
}


__device__ int GetAllInlinePoints(const int* allPoints, const  int* circleCenter, const double circleRadius, int* inlinePoints)
{
	int numOfInlinePoints = 0;
	for (int i = 0; i < NUM_OF_ALL_POINTS; ++i)
	{
		double pointRadius = Point::dist(&allPoints[2 * i], circleCenter);

		if ((pointRadius > (circleRadius - THRESHOLD_RADIUS)) && (pointRadius < (circleRadius + THRESHOLD_RADIUS)))
		{
			inlinePoints[2 * numOfInlinePoints] = allPoints[2 * i];
			inlinePoints[2 * numOfInlinePoints + 1] = allPoints[2 * i + 1];
			++numOfInlinePoints;
		}
	}

	return numOfInlinePoints;
}


__device__ double CalculateMeanError(const int* inlinePoints, const int numOfInlinePoints, const  int* circleCenter, const double circleRadius)
{
	double error = 0;

	for (int i = 0; i < numOfInlinePoints; ++i)
	{
		double pointRadius = Point::dist(&inlinePoints[2 * i], circleCenter);
		error += std::abs(circleRadius - pointRadius);
	}

	return error / numOfInlinePoints;
}


__device__ void GetCandidateCircleParameters(int* allPoints, int* circleCenter, double& circleRadius, double& error)
{
	// https://sdg002.github.io/ransac-circle/index.html
	int* chosenPoints = new int[2 * K], * inlinePoints = new int[2 * NUM_OF_ALL_POINTS];

	GetRandomPoints(allPoints, chosenPoints);

	GetCircleParametersLeastSquares(chosenPoints, N, circleCenter, circleRadius);

	int numOfInlinePoints = GetAllInlinePoints(allPoints, circleCenter, circleRadius, inlinePoints);

	if (numOfInlinePoints > THRESHOLD_COUNT)
	{
		GetCircleParametersLeastSquares(inlinePoints, numOfInlinePoints, circleCenter, circleRadius);

		numOfInlinePoints = GetAllInlinePoints(allPoints, circleCenter, circleRadius, inlinePoints);

		if (numOfInlinePoints > THRESHOLD_COUNT)
		{
			error = CalculateMeanError(inlinePoints, numOfInlinePoints, circleCenter, circleRadius);
			return;
		}
	}

	error = -1;
}


__global__ void IterateOverCandidates(int* edgesPoints, int* circleCenters, double* circleRadiuses, double* errors)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < K)
	{
		int* circleCenter = new int[2];
		double circleRadius = 0, error = -1;

		GetCandidateCircleParameters(edgesPoints, circleCenter, circleRadius, error);

		circleCenters[2 * index] = circleCenter[0];
		circleCenters[2 * index + 1] = circleCenter[1];
		circleRadiuses[index] = circleRadius;
		errors[index] = error;
	}
}


__host__ std::pair<cv::Point, double> ChooseBestParameters(int* circleCenters, double* circleRadiuses, double* errors)
{
	int bestParameterIndex = 0;
	double tempError = 10000;

	for (int i = 0; i < K; ++i)
	{
		if (errors[i] > 0 && errors[i] < tempError)
		{
			tempError = errors[i];
			bestParameterIndex = i;
		}
	}

	std::pair <cv::Point, double> circleParameters(cv::Point(circleCenters[2 * bestParameterIndex], circleCenters[2 * bestParameterIndex + 1]), circleRadiuses[bestParameterIndex]);

	// free memory
	{
		delete[] circleCenters;
		delete[] circleRadiuses;
		delete[] errors;
	}

	return circleParameters;
}


__host__ std::pair<cv::Point, double> GetOptimalParameters(int* edgesPointsCPU)
{
	int* edgesPoints, * circleCenters, * circleCentersCPU = new int[2 * K];
	double* circleRadiuses, * errors, * circleRadiusesCPU = new double[K], * errorsCPU = new double[K];

	// allocate data
	{
		cudaMallocManaged((void**)&edgesPoints, 2 * NUM_OF_ALL_POINTS * sizeof(int));
		cudaMallocManaged((void**)&circleCenters, 2 * K * sizeof(int));
		cudaMallocManaged((void**)&circleRadiuses, K * sizeof(double));
		cudaMallocManaged((void**)&errors, K * sizeof(double));

		cudaMemcpy(edgesPoints, edgesPointsCPU, 2 * NUM_OF_ALL_POINTS * sizeof(int), cudaMemcpyHostToDevice);
	}

	dim3 numBlocks(ceil(K / (double)BLOCK_SIZE));
	dim3 blockSize(BLOCK_SIZE);

	IterateOverCandidates <<<numBlocks, blockSize>>> (edgesPoints, circleCenters, circleRadiuses, errors);
	cudaDeviceSynchronize();

	// copy from gpu to cpu
	{
		cudaMemcpy(circleCentersCPU, circleCenters, 2 * K * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(circleRadiusesCPU , circleRadiuses, K * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(errorsCPU, errors, K * sizeof(double), cudaMemcpyDeviceToHost);
	}

	// free memory
	{
		cudaFree(edgesPoints);
		cudaFree(circleCenters);
		cudaFree(circleRadiuses);
		cudaFree(errors);
	}

	return ChooseBestParameters(circleCentersCPU, circleRadiusesCPU, errorsCPU);
}


void fourthTask::FindCircleGPU(const std::string pathToImage)
{
	// read data
	cv::Mat inputImage = cv::imread(pathToImage);
	cv::Mat outputImage = inputImage.clone();
	int numOfEdges = 0;

	// select data
	int* edgesPoints = GetCircleData(inputImage, numOfEdges);

	// get optimal circle parameters
	std::pair<cv::Point, double> circleParameters = GetOptimalParameters(edgesPoints);

	// add circle to the image
	{
		cv::circle(outputImage, circleParameters.first, circleParameters.second, cv::Scalar(0, 0, 0), CIRCLE_THICKNESS);
		cv::imwrite("./src/task4/OutputImage.png", outputImage);
	}
}
