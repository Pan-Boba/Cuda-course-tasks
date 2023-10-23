cmake_minimum_required(VERSION 3.27)
project(ImageProcessing LANGUAGES CUDA CXX)

add_library(ImageProcessing ImageProcessing.cuh ImageProcessing.cu)

find_package(OpenCV REQUIRED)

set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath)" ${OpenCV_INCLUDE_DIRS})
set(CMAKE_VS_SDK_LIBRARY_DIRECTORIES "$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64)" ${OpenCV_LIBS})

target_link_libraries(ImageProcessing ${OpenCV_LIBS})

file(CREATE_LINK "${PROJECT_SOURCE_DIR}/ImageSample.png" "${CMAKE_CURRENT_BINARY_DIR}/ImageSample.png" SYMBOLIC)