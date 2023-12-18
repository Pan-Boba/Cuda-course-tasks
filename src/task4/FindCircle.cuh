#pragma once

#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>
#include <numeric>
#include <functional>
#include <chrono>
#include <math.h>
#include <opencv2/opencv.hpp>

namespace fourthTask
{
	void FindCircle(std::string pathToImage);
}
