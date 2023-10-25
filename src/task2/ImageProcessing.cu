#include "ImageProcessing.cuh"

#define KERNEL_SIZE 3
#define IMAGE_CHANNELS 3
#define BLOCK_SIZE 16
#define SHARED_BLOCK_SIZE (BLOCK_SIZE - 1 + KERNEL_SIZE)
#define ITERATION_NUM 1

using namespace secondTask;

// helper-enum
static enum class MemoryType { Global = 0, Shared = 1, Texture = 2 };
inline const char* ToString(MemoryType type)
{
	switch (type)
	{
		case MemoryType::Global:	return "Global";
		case MemoryType::Shared:	return "Shared";
		case MemoryType::Texture:	return "Texture";
	}
}


static const float hostKernel[KERNEL_SIZE * KERNEL_SIZE] = { 1.0 / 16, 2.0 / 16, 1.0 / 16,
															  2.0 / 16, 4.0 / 16, 2.0 / 16,
															  1.0 / 16, 2.0 / 16, 1.0 / 16 };


__constant__ static float deviceConstantKernel[KERNEL_SIZE * KERNEL_SIZE];
static texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> deviceInputTexture;


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


__global__ static void TextureMemKernelConvolution(unsigned char* resultImage, int width, int height)
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
						sum += tex1Dfetch(deviceInputTexture, IMAGE_CHANNELS * (currentRow * width + currentCol) + channel) * deviceConstantKernel[i * KERNEL_SIZE + j];
					}
				}
			}

			resultImage[IMAGE_CHANNELS * (row * width + col) + channel] = sum;
		}
	}

}


static void BlurImageWithFixedMemoryType(const cv::Mat3b& inputImage, const unsigned char* inputData, cv::Mat3b& outputImage, unsigned char* outputData, MemoryType type)
{
	std::string outputImageName = ToString(type);
	long long totalTimeElapsed = 0;

	int imageWidth = inputImage.cols, imageHeight = inputImage.rows;
	int allocImageSize = inputImage.cols * inputImage.rows * IMAGE_CHANNELS * sizeof(unsigned char);
	
	dim3 dimGrid(ceil((float)inputImage.cols / BLOCK_SIZE), ceil((float)inputImage.rows / BLOCK_SIZE), 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

	if (type == MemoryType::Texture)
	{
		cudaChannelFormatDesc textureDescription = cudaCreateChannelDesc<unsigned int>();
		cudaBindTexture(0, deviceInputTexture, inputData, textureDescription, allocImageSize);
	}

	// mark the average time
	for (int i = 0; i < ITERATION_NUM; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();

		switch (type)
		{
			case MemoryType::Global:
			{
				GlobalMemKernelConvolution <<<dimGrid, dimBlock>>> (inputData, outputData, imageWidth, imageHeight);
				break;
			}
			case MemoryType::Shared:
			{
				SharedMemKernelConvolution <<<dimGrid, dimBlock>>> (inputData, outputData, imageWidth, imageHeight);
				break;
			}

			case MemoryType::Texture:
			{
				TextureMemKernelConvolution <<<dimGrid, dimBlock>>> (outputData, imageWidth, imageHeight);
				break;
			}
		}

		auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
		totalTimeElapsed += timeElapsed;
	}

	std::cout << "Convolution with " << outputImageName << " memory: average elapsed time " << (float) totalTimeElapsed / ITERATION_NUM << " microseconds" << std::endl;

	// Create new image
	{
		cudaMemcpy(outputImage.data, outputData, allocImageSize, cudaMemcpyDeviceToHost);
		cv::imwrite("./src/task2/Output" + outputImageName + "Memory.png", outputImage);
	}
	
	if (type == MemoryType::Texture)
	{
		cudaUnbindTexture(deviceInputTexture);
	}
}


void secondTask::Blur2DImage(std::string pathToImage)
{
	cv::Mat3b inputImage = cv::imread(pathToImage);
	cv::Mat3b ouputImage = inputImage.clone();

	int allocImageSize = inputImage.cols * inputImage.rows * IMAGE_CHANNELS * sizeof(unsigned char); 
	unsigned char* inputData, * outputData;

	// Allocate memory and copy data
	{
		cudaDeviceReset();
		cudaMallocManaged(&inputData, allocImageSize);
		cudaMallocManaged(&outputData, allocImageSize);

		cudaMemcpy(inputData, inputImage.data, allocImageSize, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(deviceConstantKernel, hostKernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
	}

	BlurImageWithFixedMemoryType(inputImage, inputData, ouputImage, outputData, MemoryType::Global);

	// Free memory
	{
		cudaFree(inputData);
		cudaFree(outputData);
		cudaFree(deviceConstantKernel);
	}
}
