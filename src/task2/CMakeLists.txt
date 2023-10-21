cmake_minimum_required(VERSION 3.27)
project(ImageProcessing LANGUAGES CUDA CXX)

set(OPENCV_INCLUDE_DIRS C:/opencv/build/include)
set(OPENCV_LIBS C:/opencv/build/x64/vc15/lib)

set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath)" ${OPENCV_INCLUDE_DIRS})
set(CMAKE_VS_SDK_LIBRARY_DIRECTORIES "$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64)" ${OPENCV_LIBS})

add_library(ImageProcessing ImageProcessing.cuh ImageProcessing.cu)
