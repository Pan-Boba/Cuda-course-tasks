cmake_minimum_required(VERSION 3.27)
project(CudaCourseTasks LANGUAGES CUDA CXX)

add_executable(CudaCourseTasks src/kernel.cu)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
set(CMAKE_SUPPRESS_REGENERATION true)

add_subdirectory(src/task1)
add_subdirectory(src/task2)

target_link_libraries(CudaCourseTasks PUBLIC ArrayAllocation)
target_link_libraries(CudaCourseTasks PUBLIC ImageProcessing)

target_include_directories(CudaCourseTasks PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/src/task1"
                          "${PROJECT_SOURCE_DIR}/src/task2"
                          )


