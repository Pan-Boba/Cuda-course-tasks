cmake_minimum_required(VERSION 3.25)
project(CudaCourseTasks LANGUAGES CUDA CXX)

add_library(ArrayAllocation ArrayAllocation.cuh ArrayAllocation.cu)
