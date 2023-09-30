#include "ArrayAllocation.cuh"

using namespace firstTask;

template <class T, class J>
/*__global__*/ static void Initialize(const int iniSize, T* arr, J(*function)(J))
{
	for (int i = 0; i < iniSize; i++)
		arr[i] = function((i % 360) * M_PI / 180.0);
}

template <class T, class J>
static double CalculateError(const int size, T* array, J(*function)(J))
{
	double error = 0.0;

	Initialize<T, J>(size, array, function);

	for (int i = 0; i < size; i++)
	{
		error += fabs(sin((i % 360) * M_PI / 180.0) - array[i]);
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
	float* floatArray = new float[arraySize];
	double* doubleArray = new double[arraySize];

	std::vector<double> errorArray
	{
		CalculateError<float, float>(arraySize, floatArray, &sinf),
		CalculateError<float, double>(arraySize, floatArray, &sin) ,
		//CalculateError<float, ? >(arraySize, floatArray, &__sinf) ,
		CalculateError<double, float>(arraySize, doubleArray, &sinf),
		CalculateError<double, double>(arraySize, doubleArray, &sin) ,
		//CalculateError<float, ? >(arraySize, doubleArray, &__sinf)
	};

	//double* doubleArray;
	//
	//// Allocate unified memory
	//cudaMallocManaged(&floatArray, arraySize * sizeof(float));
	//cudaMallocManaged(&doubleArray, arraySize * sizeof(double));
	//
	//// Run kernel on 1M elements on the GPU
	//// sin, sinf, __sinf
	//Initialize<float> <<<1, 1>>> (arraySize, floatArray, &sin);
	//Initialize<double> <<<1, 1>>> (arraySize, doubleArray, &sin);
	//
	//// Wait for GPU to finish before accessing on host
	//cudaDeviceSynchronize();

	// Check for errors
	Print(errorArray);

	// Free memory
	//cudaFree(floatArray);
	//cudaFree(doubleArray);
	delete[] floatArray;
	delete[] doubleArray;
}
