#include <iostream>
#include <string>
#include "task1\ArrayAllocation.cuh"
#include "task2\ImageProcessing.cuh"
#include "task4\FindCircle.cuh"


void RunTask(int taskNumber)
{
	switch (taskNumber)
	{
		case 1:
		{
			std::cout << "Running the first task...\n" << std::endl;
			firstTask::PrintError();
			break;
		}
		case 2:
		{
			std::cout << "Running the second task...\n" << std::endl;
			secondTask::Blur2DImage("./src/task2/ImageSample.png");
			std::cout << "Second task completed.\n" << std::endl;
			break;
		}
		case 4:
		{
			std::cout << "Running the fourth task...\n" << std::endl;
			//fourthTask::FindCircleCPU("./src/task4/InputImage.png");
			fourthTask::FindCircleCPU("./src/task4/InputImage.png");
			std::cout << "Fourth task completed.\n" << std::endl;
			break;
		}
		default:
		{
			std::cout << "Invalid number\n" << std::endl;
			break;
		}
	}
}

int main()
{
	std::string line;
	int numberOfTask;

	std::cout << "Enter task number (1, 2, 4): ";
	std::getline(std::cin, line);

	try
	{
		numberOfTask = std::stoi(line);
	}
	catch (...)
	{
		std::cout << "Invalid input" << std::endl;
		return -1;
	}

	RunTask(numberOfTask);

	system("pause");

	return 0;
}
