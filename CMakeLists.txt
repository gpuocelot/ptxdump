cmake_minimum_required(VERSION 3.1)

project(ptxdump C CXX CUDA)

# in order to retain the PTX code in the text executables,
# the target GPU architecture has to be virtual.
set(CMAKE_CUDA_ARCHITECTURES 35-virtual)

# CUDA runtime must be shared, so that we could intercept it.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Keep intermediate files produced during example compilation:
# we will use them for testing
list(APPEND CMAKE_CUDA_FLAGS "-keep")

add_library(${PROJECT_NAME} SHARED src/ptxdump.cpp src/ptxdump_preload.cpp
	ThirdParty/cuda-fatbin-decompression/fatbin-decompress.c)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/cuda-fatbin-decompression)

add_executable(${PROJECT_NAME}_example src/example.cu)

