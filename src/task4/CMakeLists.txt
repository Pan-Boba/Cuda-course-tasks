cmake_minimum_required(VERSION 3.27)
project(FindCircle LANGUAGES CUDA CXX)

add_library(FindCircle FindCircle.cuh FindCircle.cu)

find_package(OpenCV REQUIRED)

set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath)" ${OpenCV_INCLUDE_DIRS})
set(CMAKE_VS_SDK_LIBRARY_DIRECTORIES "$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64)" ${OpenCV_LIBS})

target_link_libraries(FindCircle ${OpenCV_LIBS})

file(CREATE_LINK "${PROJECT_SOURCE_DIR}/InputImage.png" "${CMAKE_CURRENT_BINARY_DIR}/InputImage.png" SYMBOLIC)