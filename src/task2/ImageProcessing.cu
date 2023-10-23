#include "ImageProcessing.cuh"

#define BlockSize 512
#define NumBlocks 100

using namespace secondTask;

const int kernel[3][3] = { 1, 2, 1,
						   2, 4, 2,
						   1, 2, 1 };

static void ConvoluteWithKernel(unsigned char* arr, unsigned char* result, int width, int height)
{
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			for (int k = 0; k < 3; k++)
			{
				int sum = 0;
				int sumKernel = 0;

				for (int j = -1; j <= 1; j++)
				{
					for (int i = -1; i <= 1; i++)
					{
						if ((row + j) >= 0 && (row + j) < height && (col + i) >= 0 && (col + i) < width)
						{
							int color = arr[(row + j) * 3 * width + (col + i) * 3 + k];
							sum += color * kernel[i + 1][j + 1];
							sumKernel += kernel[i + 1][j + 1];
						}
					}
				}

				result[3 * row * width + 3 * col + k] = sum / sumKernel;
			}
		}
	}
}

void secondTask::Blur2DImage(std::string pathToImage)
{
	cv::Mat3b image = cv::imread(pathToImage);
	cv::Mat3b result = image.clone();

	ConvoluteWithKernel(image.data, result.data, image.cols, image.rows);

	cv::imwrite("./src/task2/ProcessedImage.png", result);
}
