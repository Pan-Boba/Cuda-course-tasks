#pragma once

#include "cuda_runtime.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <math.h>
#include <opencv2/opencv.hpp>

namespace secondTask
{
	void Blur2DImage(std::string pathToImage);
}
