#include "ImageProcessing.cuh"

#define KERNEL_SIZE 3
#define IMAGE_CHANNELS 3
#define BLOCK_SIZE 16
#define SHARED_BLOCK_SIZE (BLOCK_SIZE - 1 + KERNEL_SIZE)

using namespace secondTask;

static const float hostKernel[KERNEL_SIZE * KERNEL_SIZE] = { 1.0 / 16, 2.0 / 16, 1.0 / 16,
															  2.0 / 16, 4.0 / 16, 2.0 / 16,
															  1.0 / 16, 2.0 / 16, 1.0 / 16 };

// __constant__ static float deviceConstantInputData[892 * 460 * IMAGE_CHANNELS]; - wont'do: too large image
__constant__ static float deviceConstantKernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ static void GlobalMemKernelConvolution(const unsigned char* inputImage, unsigned char* resultImage, int width, int height)
{
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

	for (int channel = 0; channel < IMAGE_CHANNELS; channel++)
	{
		if (row < height && col < width) {
			unsigned int sum = 0;


			for (int i = 0; i < KERNEL_SIZE; i++)
			{
				for (int j = 0; j < KERNEL_SIZE; j++)
				{
					unsigned int currentRow = row - 1 + i;
					unsigned int currentCol = col - 1 + j;

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

		unsigned int destCol = (threadIdx.y * BLOCK_SIZE + threadIdx.x) / SHARED_BLOCK_SIZE;
		unsigned int destRow = (threadIdx.y * BLOCK_SIZE + threadIdx.x) % SHARED_BLOCK_SIZE;
		unsigned int srcCol = blockIdx.y * BLOCK_SIZE + destCol - KERNEL_SIZE / 2;
		unsigned int srcRow = blockIdx.x * BLOCK_SIZE + destRow - KERNEL_SIZE / 2;

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
		unsigned int sum = 0;
		for (int i = 0; i < KERNEL_SIZE; i++)
		{
			for (int j = 0; j < KERNEL_SIZE; j++)
			{
				sum += sharedInputImageBlock[threadIdx.y + i][threadIdx.x + j] * deviceConstantKernel[i * KERNEL_SIZE + j];
			}
		}

		unsigned int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
		unsigned int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if (row < height && col < width)
		{
			resultImage[IMAGE_CHANNELS * (row * width + col) + channel] = sum;
		}

		__syncthreads();
	}
}


__global__ static void TextureMemKernelConvolution(const cudaTextureObject_t inputImageTexture, unsigned char* resultImage, int width, int height)
{
	// TO DO
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

	for (int channel = 0; channel < IMAGE_CHANNELS; channel++)
	{
		if (row < height && col < width) {
			unsigned int sum = 0;


			for (int i = 0; i < KERNEL_SIZE; i++)
			{
				for (int j = 0; j < KERNEL_SIZE; j++)
				{
					unsigned int currentRow = row - 1 + i;
					unsigned int currentCol = col - 1 + j;

					if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width)
					{
						sum += tex1D<int>(inputImageTexture, IMAGE_CHANNELS * (currentRow * width + currentCol) + channel) * deviceConstantKernel[i * KERNEL_SIZE + j];
					}
				}
			}

			resultImage[IMAGE_CHANNELS * (row * width + col) + channel] = sum;
		}
	}

}



static void BlurImageWithGlobalMemory(const cv::Mat3b& inputImage, int imageWidth, int imageHeight)
{
	cv::Mat3b outputImage = inputImage.clone();

	int allocImageSize = imageWidth * imageHeight * IMAGE_CHANNELS * sizeof(unsigned char);
	unsigned char* inputData, * outputData;

	// Allocate memory and copy data
	{
		cudaMallocManaged(&inputData, allocImageSize);
		cudaMallocManaged(&outputData, allocImageSize);

		cudaMemcpy(inputData, inputImage.data, allocImageSize, cudaMemcpyHostToDevice);
	}

	dim3 dimGrid(ceil((float)imageWidth / BLOCK_SIZE), ceil((float)imageHeight / BLOCK_SIZE), 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

	{
		auto start = std::chrono::high_resolution_clock::now();

		GlobalMemKernelConvolution <<<dimGrid, dimBlock>>> (inputData, outputData, imageWidth, imageHeight);

		auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
		std::cout << "Convolution with global memory: elapsed time " << timeElapsed << " microseconds" << std::endl;
	}

	// Create new image
	cudaMemcpy(outputImage.data, outputData, allocImageSize, cudaMemcpyDeviceToHost);
	cv::imwrite("./src/task2/OutputGlobalMemory.png", outputImage);

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
	//// TO DO
	//cv::Mat3b outputImage = inputImage.clone();
	//cudaArray_t inputTextureData;
	//
	//int allocImageSize = imageWidth * imageHeight * IMAGE_CHANNELS * sizeof(unsigned char);
	//unsigned char* * outputData;
	//
	//// Allocate memory and copy data
	//{
	//	cudaDeviceReset();
	//
	//	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	//
	//	cudaMallocManaged(&inputTextureData, allocImageSize);
	//	cudaMallocManaged(&outputData, allocImageSize);
	//
	//	//cudaMemcpy(inputData, inputImage.data, allocImageSize, cudaMemcpyHostToDevice);
	//}
	//
	//dim3 dimGrid(ceil((float)imageWidth / BLOCK_SIZE), ceil((float)imageHeight / BLOCK_SIZE), 1);
	//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	//
	//{
	//	auto start = std::chrono::high_resolution_clock::now();
	//
	//	TextureMemKernelConvolution <<<dimGrid, dimBlock>>> (inputData, outputData, imageWidth, imageHeight);
	//
	//	auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
	//	std::cout << "Convolution with texture memory: elapsed time " << timeElapsed << " microseconds" << std::endl;
	//}
	//
	//// Create new image
	//cudaMemcpy(outputImage.data, outputData, allocImageSize, cudaMemcpyDeviceToHost);
	//cv::imwrite("./src/task2/OutputTextureMemory.png", outputImage);
	//
	//// Free memory
	//cudaFree(outputData);
}


void secondTask::Blur2DImage(std::string pathToImage)
{
	cv::Mat3b inputImage = cv::imread(pathToImage);

	cudaDeviceReset();
	// special for const mem
	cudaMemcpyToSymbol(deviceConstantKernel, hostKernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

	BlurImageWithGlobalMemory(inputImage, inputImage.cols, inputImage.rows);
	BlurImageWithSharedMemory(inputImage, inputImage.cols, inputImage.rows);
	//BlurImageWithTextureMemory(inputImage, inputImage.cols, inputImage.rows);

	cudaFree(deviceConstantKernel);
}
