#include "ImageProcessing.cuh"

#define KERNEL_SIZE 3
#define IMAGE_CHANNELS 3
#define BLOCK_SIZE 16
#define SHARED_BLOCK_SIZE (BLOCK_SIZE - 1 + KERNEL_SIZE)

using namespace secondTask;

const float hostKernel[KERNEL_SIZE * KERNEL_SIZE] = { 1.0 / 16, 2.0 / 16, 1.0 / 16,
													  2.0 / 16, 4.0 / 16, 2.0 / 16,
													  1.0 / 16, 2.0 / 16, 1.0 / 16 };

// __constant__ float deviceConstantInputData[892 * 460 * 3]; - wont'do: too large image
__constant__ float deviceConstantKernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ static void ConstantMemKernelConvolution(const unsigned char* inputImage, unsigned char* resultImage, int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	for (int channel = 0; channel < IMAGE_CHANNELS; channel++)
	{
		if (row < height && col < width) {
			int sum = 0;


			for (int i = 0; i < KERNEL_SIZE; i++)
			{
				for (int j = 0; j < KERNEL_SIZE; j++)
				{
					int currentRow = row - 1 + i;
					int currentCol = col - 1 + j;

					if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width)
					{
						sum += inputImage[IMAGE_CHANNELS * (currentRow * width + currentCol) + channel] * deviceConstantKernel[i * KERNEL_SIZE + j];
					}
				}
			}

			resultImage[IMAGE_CHANNELS * (row * width + col) + channel] = sum;
		}
	}
}


__global__ static void SharedMemKernelConvolution(const unsigned char* inputImage, unsigned char* resultImage, int width, int height)
{
	__shared__ unsigned char sharedInputImageBlock[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

	for (int channel = 0; channel < IMAGE_CHANNELS; channel++)
	{
		// https://stackoverflow.com/questions/21380549/upload-data-in-shared-memory-for-convolution-kernel

		int destCol = (threadIdx.y * BLOCK_SIZE + threadIdx.x) / SHARED_BLOCK_SIZE;
		int destRow = (threadIdx.y * BLOCK_SIZE + threadIdx.x) % SHARED_BLOCK_SIZE;
		int srcCol = blockIdx.y * BLOCK_SIZE + destCol - KERNEL_SIZE / 2;
		int srcRow = blockIdx.x * BLOCK_SIZE + destRow - KERNEL_SIZE / 2;

		if (srcCol >= 0 && srcCol < height && srcRow >= 0 && srcRow < width)
		{
			sharedInputImageBlock[destCol][destRow] = inputImage[IMAGE_CHANNELS * (srcCol * width + srcRow) + channel];
		}
		else
		{
			sharedInputImageBlock[destCol][destRow] = 0;
		}

		destCol = (threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE) / SHARED_BLOCK_SIZE;
		destRow = (threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE) % SHARED_BLOCK_SIZE;
		srcCol = blockIdx.y * BLOCK_SIZE + destCol - KERNEL_SIZE / 2;
		srcRow = blockIdx.x * BLOCK_SIZE + destRow - KERNEL_SIZE / 2;

		if (srcCol >= 0 && srcCol < height && srcRow >= 0 && srcRow < width)
		{
			sharedInputImageBlock[destCol][destRow] = inputImage[IMAGE_CHANNELS * (srcCol * width + srcRow) + channel];
		}
		else
		{
			sharedInputImageBlock[destCol][destRow] = 0;
		}

		__syncthreads();

		// kernel convolution
		int sum = 0;
		for (int i = 0; i < KERNEL_SIZE; i++)
		{
			for (int j = 0; j < KERNEL_SIZE; j++)
			{
				sum += sharedInputImageBlock[threadIdx.y + i][threadIdx.x + j] * deviceConstantKernel[i * KERNEL_SIZE + j];
			}
		}

		int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
		int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if (row < height && col < width)
		{
			resultImage[IMAGE_CHANNELS * (row * width + col) + channel] = sum;
		}

		__syncthreads();
	}
}


__global__ static void TextureMemKernelConvolution(unsigned char* resultImage, int width, int height)
{
	// TO DO

}



static void BlurImageWithGlobalMemory(const cv::Mat3b& inputImage, int imageWidth, int imageHeight)
{
	cv::Mat3b outputImage = inputImage.clone();

	int allocImageSize = imageWidth * imageHeight * IMAGE_CHANNELS * sizeof(unsigned char);
	unsigned char* inputData, * outputData;

	// Allocate memory and copy data
	{
		cudaDeviceReset();

		cudaMallocManaged(&inputData, allocImageSize);
		cudaMallocManaged(&outputData, allocImageSize);

		cudaMemcpy(inputData, inputImage.data, allocImageSize, cudaMemcpyHostToDevice);
	}

	dim3 dimGrid(ceil((float)imageWidth / BLOCK_SIZE), ceil((float)imageHeight / BLOCK_SIZE), 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

	{
		auto start = std::chrono::high_resolution_clock::now();

		ConstantMemKernelConvolution <<<dimGrid, dimBlock>>> (inputData, outputData, imageWidth, imageHeight);

		auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
		std::cout << "Convolution with constant memory: elapsed time " << timeElapsed << " microseconds" << std::endl;
	}

	// Create new image
	cudaMemcpy(outputImage.data, outputData, allocImageSize, cudaMemcpyDeviceToHost);
	cv::imwrite("./src/task2/OutputConstMemory.png", outputImage);

	// Free memory
	cudaFree(inputData);
	cudaFree(outputData);
}


static void BlurImageWithSharedMemory(const cv::Mat3b& inputImage, int imageWidth, int imageHeight)
{
	cv::Mat3b outputImage = inputImage.clone();

	int allocImageSize = imageWidth * imageHeight * IMAGE_CHANNELS * sizeof(unsigned char);
	unsigned char* inputData, * outputData;

	// Allocate memory and copy data
	{
		cudaDeviceReset();

		cudaMallocManaged(&inputData, allocImageSize);
		cudaMallocManaged(&outputData, allocImageSize);

		cudaMemcpy(inputData, inputImage.data, allocImageSize, cudaMemcpyHostToDevice);
	}

	dim3 dimGrid(ceil((float)imageWidth / BLOCK_SIZE), ceil((float)imageHeight / BLOCK_SIZE), 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

	{
		auto start = std::chrono::high_resolution_clock::now();

		SharedMemKernelConvolution <<<dimGrid, dimBlock>>> (inputData, outputData, imageWidth, imageHeight);

		auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
		std::cout << "Convolution with shared memory: elapsed time " << timeElapsed << " microseconds" << std::endl;
	}

	// Create new image
	cudaMemcpy(outputImage.data, outputData, allocImageSize, cudaMemcpyDeviceToHost);
	cv::imwrite("./src/task2/OutputSharedMemory.png", outputImage);

	// Free memory
	cudaFree(inputData);
	cudaFree(outputData);
}


static void BlurImageWithTextureMemory(const cv::Mat3b& inputImage, int imageWidth, int imageHeight)
{
	// TO DO
}


void secondTask::Blur2DImage(std::string pathToImage)
{
	int allocKernelSize = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
	cv::Mat3b inputImage = cv::imread(pathToImage);

	// special for const mem
	cudaMemcpyToSymbol(deviceConstantKernel, hostKernel, allocKernelSize);

	BlurImageWithGlobalMemory(inputImage, inputImage.cols, inputImage.rows);
	BlurImageWithSharedMemory(inputImage, inputImage.cols, inputImage.rows);
	BlurImageWithTextureMemory(inputImage, inputImage.cols, inputImage.rows);

	cudaFree(deviceConstantKernel);
}
