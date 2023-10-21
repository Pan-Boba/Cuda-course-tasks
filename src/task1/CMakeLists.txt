cmake_minimum_required(VERSION 3.25)

project(ArrayAllocation LANGUAGES CUDA CXX)

add_library(ArrayAllocation ArrayAllocation.cuh ArrayAllocation.cu)
