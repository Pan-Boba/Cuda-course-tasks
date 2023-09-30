cmake_minimum_required(VERSION 3.25)
project(CudaCourseTasks LANGUAGES CUDA CXX)

set(CMAKE_SUPPRESS_REGENERATION true)

add_executable(CudaCourseTasks src/kernel.cu)

add_subdirectory(src/task1)
target_link_libraries(CudaCourseTasks PUBLIC ArrayAllocation)

target_include_directories(CudaCourseTasks PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/src/task1"
                          )

