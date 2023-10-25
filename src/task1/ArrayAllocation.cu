#include "ArrayAllocation.cuh"

#define BlockSize 512
#define NumBlocks 100

using namespace firstTask;

static enum class Function { sin, sinf, __sinf };

#pragma region Initialize

template <class T>
__global__ static void InitializeWithSin(const int iniSize, T* arr)
{
	unsigned int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < iniSize; i += stride)
	{
		arr[i] = sin((i % 360) * M_PI / 180.0);
	}
}

template <class T>
__global__ static void InitializeWithSinf(const int iniSize, T* arr)
{
	unsigned int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < iniSize; i += stride)
	{
		arr[i] = sinf((i % 360) * M_PI / 180.0);
	}
}

template <class T>
__global__ static void InitializeWithFastSin(const int iniSize, T* arr)
{
	unsigned int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < iniSize; i += stride)
	{
		arr[i] = __sinf((i % 360) * M_PI / 180.0);
	}
}


#pragma endregion

template <class T>
static std::pair< double, double > CalculateError(const int size, T* arr, Function mode)
{
	dim3 blockSize(BlockSize);
	dim3 numBlocks(NumBlocks);
	double error = 0.0;
	auto start = std::chrono::high_resolution_clock::now();

	switch (mode)
	{
		case Function::sin:
		{
			InitializeWithSin<T> << <numBlocks, blockSize >> > (size, arr);
			break;
		}
		case Function::sinf:
		{
			InitializeWithSinf<T> << <numBlocks, blockSize >> > (size, arr);
			break;
		}

		case Function::__sinf:
		{
			InitializeWithFastSin<T> << <numBlocks, blockSize >> > (size, arr);
			break;
		}
	}

	// Wait for GPU to finish
	cudaDeviceSynchronize();

	auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

	for (int i = 0; i < size; i++)
	{
		error += fabs(sin((i % 360) * M_PI / 180.0) - arr[i]);
	}

	return { error / 1e8, timeElapsed };
}

static void Print(const std::vector<std::pair< double, double >>& array)
{
	std::cout << "Errors for float array (sinf, sin, __sinf): ";

	for (int i = 0; i < array.size(); i++)
	{
		std::cout << array.at(i).first << " (Time: " << array.at(i).second << " microseconds); ";

		if (i == array.size() / 2 - 1)
		{
			std::cout << "\nErrors for double array (sinf, sin, __sinf): ";
		}
	}

	std::cout << "\n\n";
}

void firstTask::PrintError()
{
	//Function
	int arraySize = 1e8;
	float* floatArray;
	double* doubleArray;

	// Allocate unified memory
	cudaMallocManaged(&floatArray, arraySize * sizeof(float));
	cudaMallocManaged(&doubleArray, arraySize * sizeof(double));

	std::vector < std::pair< double, double >> errorArray
	{
		CalculateError<float>(arraySize, floatArray, Function::sinf),
		CalculateError<float>(arraySize, floatArray, Function::sin),
		CalculateError<float>(arraySize, floatArray, Function::__sinf),

		CalculateError<double>(arraySize, doubleArray, Function::sinf),
		CalculateError<double>(arraySize, doubleArray, Function::sin),
		CalculateError<double>(arraySize, doubleArray, Function::__sinf)
	};

	// Check for errors
	Print(errorArray);

	// Free memory
	cudaFree(floatArray);
	cudaFree(doubleArray);
}
