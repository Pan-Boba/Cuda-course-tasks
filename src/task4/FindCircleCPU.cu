#include "FindCircle.cuh"

#define BLUR_SIZE 17
#define CIRCLE_THICKNESS 3
#define N 5
#define K 100
#define THRESHOLD_RADIUS 5
#define THRESHOLD_COUNT 900 //100 for inner circle

using namespace fourthTask;

std::vector<cv::Point> GetCircleData(const cv::Mat& inputImage)
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

	// Create vector of edge Points
	std::vector<cv::Point> edgesVector;

	for (int row = 0; row < edges.rows; ++row)
	{
		for (int col = 0; col < edges.cols; ++col)
		{
			if (edges.data[row * edges.cols + col] == 255)
			{
				edgesVector.emplace_back(cv::Point(col, row));
			}
		}
	}

	return edgesVector;
}


std::vector<cv::Point> GetRandomPoints(std::vector<cv::Point> edgesVector)
{
	std::random_shuffle(edgesVector.begin(), edgesVector.end());
	std::vector<cv::Point> selectedEdges(edgesVector.begin(), edgesVector.begin() + N);
	return selectedEdges;
}


void GetCircleParametersLeastSquares(const std::vector<cv::Point>& points, cv::Point& circleCenter, double& circleRadius)
{
	// https://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf

	int pointsCount = points.size();
	cv::Mat ui(1, pointsCount, CV_32FC1, 0.0f), vi(1, pointsCount, CV_32FC1, 0.0f);

	for (int i = 0; i < pointsCount; ++i)
	{
		ui.at<float>(0, i) = (float) points.at(i).x;
		vi.at<float>(0, i) = (float) points.at(i).y;
	}

	double averageX = cv::sum(ui)[0] / (double) pointsCount;
	double averageY = cv::sum(vi)[0] / (double) pointsCount;

	cv::subtract(ui, averageX, ui);
	cv::subtract(vi, averageY, vi);

	double Suu = cv::sum(ui.mul(ui))[0];
	double Svv = cv::sum(vi.mul(vi))[0];
	double Suv = cv::sum(ui.mul(vi))[0];

	double Suuu = cv::sum(ui.mul(ui).mul(ui))[0];
	double Svvv = cv::sum(vi.mul(vi).mul(vi))[0];
	double Suvv = cv::sum(ui.mul(vi).mul(vi))[0];
	double Svuu = cv::sum(vi.mul(ui).mul(ui))[0];

	double rightSideArray[2] = { 0.5 * (Suuu + Suvv), 0.5 * (Svvv + Svuu) };
	double leftSideArray[2][2] = { {Suu, Suv}, {Suv, Svv} };

	cv::Mat rightSideVector(2, 1, CV_64FC1, rightSideArray);
	cv::Mat leftSideMatrix(2, 2, CV_64FC1, leftSideArray);

	cv::Mat centerVector = leftSideMatrix.inv() * rightSideVector;

	double dataX = centerVector.at<double>(0, 0), dataY = centerVector.at<double>(1, 0);

	circleCenter = cv::Point((int) (dataX + averageX), (int) (dataY + averageY));
	circleRadius = (int) std::sqrt(dataX * dataX + dataY * dataY + (Suu + Svv) / pointsCount);
}


std::vector<cv::Point> GetAllInlinePoints(const std::vector<cv::Point>& allPoints, const cv::Point& circleCenter, const double& circleRadius)
{
	std::vector<cv::Point> inlinePoints;

	for (int j = 0; j < allPoints.size(); ++j)
	{
		double pointRadius = cv::norm(allPoints.at(j) - circleCenter);

		if ((pointRadius > (circleRadius - THRESHOLD_RADIUS)) && (pointRadius < (circleRadius + THRESHOLD_RADIUS)))
		{
			inlinePoints.emplace_back(allPoints.at(j));
		}
	}

	return inlinePoints;
}


double CalculateMeanError(const std::vector<cv::Point>& inlinePoints, const cv::Point& circleCenter, const double& circleRadius)
{
	double error = 0;

	for (int j = 0; j < inlinePoints.size(); ++j)
	{
		double pointRadius = cv::norm(inlinePoints.at(j) - circleCenter);
		error += std::abs(circleRadius - pointRadius);
	}

	return error / inlinePoints.size();
}


void GetCandidateCircleParameters(const std::vector<cv::Point>& allPoints, cv::Point& circleCenter, double& circleRadius, double& error)
{
	// https://sdg002.github.io/ransac-circle/index.html
	std::vector<cv::Point> chosenEdges = GetRandomPoints(allPoints);

	GetCircleParametersLeastSquares(chosenEdges, circleCenter, circleRadius);

	std::vector<cv::Point> inlinePoints = GetAllInlinePoints(allPoints, circleCenter, circleRadius);

	if (inlinePoints.size() > THRESHOLD_COUNT) 
	{
		GetCircleParametersLeastSquares(inlinePoints, circleCenter, circleRadius);

		inlinePoints = GetAllInlinePoints(allPoints, circleCenter, circleRadius);

		if (inlinePoints.size() > THRESHOLD_COUNT)
		{
			error = CalculateMeanError(inlinePoints, circleCenter, circleRadius);
			return;
		}
	}

	error = -1;
}


std::pair<cv::Point, double> ChooseBestParameters(const std::vector<cv::Point>& circleCenters, const std::vector<double>& circleRadiuses, const std::vector<double>& errors)
{
	assert(!errors.empty());

	int bestParameterIndex = 0;

	double tempError = 10000;

	for (int i = 0; i < errors.size(); ++i)
	{
		if (errors.at(i) < tempError)
		{
			tempError = errors.at(i);
			bestParameterIndex = i;
		}
	}

	std::cout << circleCenters.at(bestParameterIndex) << " " << circleRadiuses.at(bestParameterIndex) << std::endl;
	std::pair <cv::Point, double> circleParameters(circleCenters.at(bestParameterIndex), circleRadiuses.at(bestParameterIndex));

	return circleParameters;
}


void fourthTask::FindCircleCPU(const std::string pathToImage)
{
	// read data
	cv::Mat inputImage = cv::imread(pathToImage);
	cv::Mat outputImage = inputImage.clone();
	std::vector<cv::Point> circleCenters;
	std::vector<double> circleRadiuses, errors;

	// select data
	std::vector<cv::Point> edgesVector = GetCircleData(inputImage);

	for (int i = 0; i < K; ++i)
	{
		cv::Point circleCenter;
		double circleRadius, error;

		GetCandidateCircleParameters(edgesVector, circleCenter, circleRadius, error);

		if (error != -1)
		{
			circleCenters.emplace_back(circleCenter);
			circleRadiuses.emplace_back(circleRadius);
			errors.emplace_back(error);
		}
	}

	std::pair<cv::Point, double> circleParameters = ChooseBestParameters(circleCenters, circleRadiuses, errors);

	// add circle to the image
	cv::circle(outputImage, circleParameters.first, circleParameters.second, cv::Scalar(0, 0, 0), CIRCLE_THICKNESS);
	cv::imwrite("./src/task4/OutputImage.png", outputImage);
}
