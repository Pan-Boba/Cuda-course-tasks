#include "ArrayAllocation.cuh"

using namespace firstTask;

template <class T, class J>
__global__ static void Initialize(const int iniSize, T* arr, J(*function)(J))
{
	unsigned int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < iniSize; i += stride)
	{
		arr[i] = function((i % 360) * M_PI / 180.0);
	}
}

template <class T, class J>
static double CalculateError(const int size, T* arr, J(*function)(J))
{
	double error = 0.0;
	dim3 blockSize(512);
	dim3 numBlocks(100);

	Initialize<T, J><<<numBlocks, blockSize>>>(size, arr, function);

	// Wait for GPU to finish
	cudaDeviceSynchronize();

	for (int i = 0; i < size; i++)
	{
		error += fabs(sin((i % 360) * M_PI / 180.0) - arr[i]);
	}

	return error;
}

static void Print(const std::vector<double>& array)
{
	std::cout << "Errors for float array (sinf, (double) sin): ";

	for (int i = 0; i < array.size(); i++)
	{
		std::cout << array.at(i) << "; ";

		if (i == array.size() / 2 - 1)
		{
			std::cout << "\nErrors for double array (sinf, (double) sin): ";
		}
	}

	std::cout << "\n\n";
}

void firstTask::PrintError()
{
	int arraySize = 1e8;
	float* floatArray;
	double* doubleArray;

	// Allocate unified memory
	cudaMallocManaged(&floatArray, arraySize * sizeof(float));
	cudaMallocManaged(&doubleArray, arraySize * sizeof(double));

	std::vector<double> errorArray
	{
		CalculateError<float, float>(arraySize, floatArray, &sinf),
		CalculateError<float, double>(arraySize, floatArray, &sin),
		//CalculateError<float, float>(arraySize, floatArray, &__sinf),
		CalculateError<double, float>(arraySize, doubleArray, &sinf),
		CalculateError<double, double>(arraySize, doubleArray, &sin),
		//CalculateError<float, float>(arraySize, doubleArray, &__sinf)
	};

	// Check for errors
	Print(errorArray);

	// Free memory
	cudaFree(floatArray);
	cudaFree(doubleArray);
}
