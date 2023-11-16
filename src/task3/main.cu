#include <omp.h>
#include <png.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <math.h>
#include <cstring>


#define KERNEL_SIZE 3
#define IMAGE_CHANNELS 3
#define BLOCK_SIZE 16
#define SHARED_BLOCK_SIZE (BLOCK_SIZE - 1 + KERNEL_SIZE)
#define ITERATION_NUM 3


bool loadImage(const char* filename, png_bytep** row_pointers, int& width, int& height)
{
	FILE* file = nullptr;
	png_structp png = nullptr;
	png_infop info = nullptr;

	try
	{
		file = fopen(filename, "rb");

		png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

		info = png_create_info_struct(png);

		png_init_io(png, file);
		png_read_info(png, info);

		width = png_get_image_width(png, info);
		height = png_get_image_height(png, info);

		*row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
		for (int i = 0; i < height; ++i)
		{
			(*row_pointers)[i] = (png_byte*)malloc(sizeof(png_byte) * width * IMAGE_CHANNELS);
		}

		png_read_image(png, (*row_pointers));

		png_destroy_read_struct(&png, &info, nullptr);
		fclose(file);

		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;

		if (file) fclose(file);
		if (png) png_destroy_read_struct(&png, nullptr, nullptr);
		if (info) png_destroy_read_struct(nullptr, &info, nullptr);

		return false;
	}
}

bool saveImage(const char* filename, png_bytep** row_pointers, const int& width, const int& height)
{
	FILE* file = nullptr;
	png_structp png = nullptr;
	png_infop info = nullptr;

	try
	{
		file = fopen(filename, "wb");

		png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

		info = png_create_info_struct(png);

		png_init_io(png, file);

		png_set_IHDR(png, info, width, height, 8,
			PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
		png_write_info(png, info);

		png_write_image(png, (*row_pointers));
		png_write_end(png, info);

		png_destroy_write_struct(&png, &info);
		fclose(file);

		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;

		if (file) fclose(file);
		if (png) png_destroy_read_struct(&png, nullptr, nullptr);

		return false;
	}
}


// helper-enum
enum class MemoryType { Global = 0, Shared = 1, Texture = 2 };
inline const char* ToString(MemoryType type)
{
	switch (type)
	{
	case MemoryType::Global:	return "Global";
	case MemoryType::Shared:	return "Shared";
	case MemoryType::Texture:	return "Texture";
	default:      return "Unknown";
	}
}


const float hostKernel[KERNEL_SIZE * KERNEL_SIZE] = { 1.0 / 16, 2.0 / 16, 1.0 / 16,
															  2.0 / 16, 4.0 / 16, 2.0 / 16,
															  1.0 / 16, 2.0 / 16, 1.0 / 16 };


__constant__ float deviceConstantKernel[KERNEL_SIZE * KERNEL_SIZE];
texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> deviceInputTexture;


__global__ void GlobalMemKernelConvolution(const unsigned char* inputImage, unsigned char* resultImage, int width, int height, int startRow)
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
					unsigned int currentRow = row - 1 + i + startRow;
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


__global__ void SharedMemKernelConvolution(const unsigned char* inputImage, unsigned char* resultImage, int width, int height, int startRow)
{
	__shared__ unsigned char sharedInputImageBlock[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

	for (int channel = 0; channel < IMAGE_CHANNELS; channel++)
	{
		// https://stackoverflow.com/questions/21380549/upload-data-in-shared-memory-for-convolution-kernel

		unsigned int destCol = (threadIdx.y * BLOCK_SIZE + threadIdx.x) / SHARED_BLOCK_SIZE;
		unsigned int destRow = (threadIdx.y * BLOCK_SIZE + threadIdx.x) % SHARED_BLOCK_SIZE;
		unsigned int srcCol = blockIdx.y * BLOCK_SIZE + destCol - KERNEL_SIZE / 2 + startRow;
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
		srcCol = blockIdx.y * BLOCK_SIZE + destCol - KERNEL_SIZE / 2 + startRow;
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


__global__ void TextureMemKernelConvolution(unsigned char* resultImage, int width, int height, int startRow)
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
					unsigned int currentRow = row - 1 + i + startRow;
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


void BlurImageWithFixedMemoryType(const unsigned char* inputData, unsigned char* outputData, int allocImageSize, int imageWidth, int imageHeight, MemoryType type)
{
	std::string outputImageName = ToString(type);
	int GPUnum;

	cudaGetDeviceCount(&GPUnum);
	omp_set_num_threads(GPUnum);

	unsigned char outputImageLocal[GPUnum][allocImageSize / GPUnum];
	#pragma omp parallel
	{
		long long totalTimeElapsed = 0;
		unsigned char* tempInputData, * tempOutputData;

		int cpu_thread_id = omp_get_thread_num();
		int num_cpu_threads = omp_get_num_threads();
		int startRow = cpu_thread_id * imageHeight / num_cpu_threads;
		cudaSetDevice(cpu_thread_id % num_cpu_threads);

		// All mem
		{
			cudaMallocManaged(&tempInputData, allocImageSize);
			cudaMallocManaged(&tempOutputData, allocImageSize / num_cpu_threads);
			cudaMemcpyToSymbol(deviceConstantKernel, hostKernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
			cudaMemcpy(tempInputData, inputData, allocImageSize, cudaMemcpyHostToDevice);
		
			if (type == MemoryType::Texture)
			{
				cudaBindTexture(0, deviceInputTexture, tempInputData, allocImageSize);
			}
		}

		dim3 dimGrid(ceil((float)imageWidth / BLOCK_SIZE), ceil((float)imageHeight / BLOCK_SIZE / num_cpu_threads), 1);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

		// mark the average time
		for (int i = 0; i < ITERATION_NUM; i++)
		{
			auto start = std::chrono::high_resolution_clock::now();

			switch (type)
			{
			case MemoryType::Global:
			{
				GlobalMemKernelConvolution << <dimGrid, dimBlock >> > (tempInputData, tempOutputData, imageWidth, imageHeight, startRow);
				break;
			}
			case MemoryType::Shared:
			{
				SharedMemKernelConvolution << <dimGrid, dimBlock >> > (tempInputData, tempOutputData, imageWidth, imageHeight, startRow);
				break;
			}

			case MemoryType::Texture:
			{
				TextureMemKernelConvolution << <dimGrid, dimBlock >> > (tempOutputData, imageWidth, imageHeight, startRow);
				break;
			}
			}

			cudaDeviceSynchronize();

			auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
			totalTimeElapsed += timeElapsed;
		}

		std::cout << "Convolution with " << outputImageName << " memory: average elapsed time " << (float)totalTimeElapsed / ITERATION_NUM << " microseconds" << std::endl;

		// Create new image
		cudaMemcpy(outputImageLocal[cpu_thread_id], tempOutputData, allocImageSize / num_cpu_threads, cudaMemcpyDeviceToHost);

		// Free mem
		{
			if (type == MemoryType::Texture)
			{
				cudaUnbindTexture(deviceInputTexture);
			}

			cudaFree(tempInputData);
			cudaFree(tempOutputData);
		}
	}

	cudaDeviceSynchronize();

	for (int i = 0; i < GPUnum; ++i)
	{
		size_t offset = i * (allocImageSize / GPUnum);

		std::memcpy(outputData + offset, outputImageLocal[i], allocImageSize / GPUnum);
	}
}


int main()
{
	int width, height;
	png_bytep* row_pointers;

	if (!loadImage("ImageSample.png", &row_pointers, width, height))
	{
		return -1;
	}

	int allocImageSize = width * height * IMAGE_CHANNELS * sizeof(unsigned char);
	auto* inputData = new unsigned char[allocImageSize];
	auto* outputData = new unsigned char[allocImageSize];

	for (int i = 0; i < height; i++)
	{
		memcpy(&inputData[i * width * IMAGE_CHANNELS * sizeof(unsigned char)], row_pointers[i], width * IMAGE_CHANNELS * sizeof(unsigned char));
	}

	for (int i = 0; i < 3; ++i)
	{
		BlurImageWithFixedMemoryType(inputData, outputData, allocImageSize, width, height, (MemoryType)i);

		for (int j = 0; j < height; j++)
		{
			memcpy(row_pointers[j], &outputData[j * width * IMAGE_CHANNELS * sizeof(unsigned char)], width * IMAGE_CHANNELS * sizeof(unsigned char));
		}

		std::string imageName = ToString((MemoryType)i);
		std::string outputFileName = imageName + "Memory.png";
		if (!saveImage(outputFileName.c_str(), &row_pointers, width, height))
		{
			return -1;
		}
	}

	// Free memory
	{
		cudaFree(deviceConstantKernel);

		delete[] inputData;
		delete[] outputData;

		for (int i = 0; i < height; ++i)
		{
			delete[] row_pointers[i];
		}

		delete[] row_pointers;
	}

	return 0;
}
