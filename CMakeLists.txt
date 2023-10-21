cmake_minimum_required(VERSION 3.27)
project(CudaCourseTasks LANGUAGES CUDA CXX)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
set(CMAKE_SUPPRESS_REGENERATION true)

add_subdirectory(src/task1)
add_subdirectory(src/task2)
add_executable(CudaCourseTasks src/kernel.cu)
