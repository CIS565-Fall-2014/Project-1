cmake_minimum_required(VERSION 2.8)
project(565matrix)

find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

list(APPEND SOURCES
    "src/matrix_math.cu"
    )

message(STATUS "SOURCES:" ${SOURCES})

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang|GNU")
    list(APPEND CMAKE_CXX_FLAGS "-Wall")
    list(APPEND CMAKE_CXX_FLAGS_RELEASE "-O3 -mtune=native -ffast-math")
    list(APPEND CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

cuda_add_executable(${CMAKE_PROJECT_NAME} ${SOURCES} OPTIONS -arch=sm_20)
#target_link_libraries(${CMAKE_PROJECT_NAME} ${CORELIBS})
