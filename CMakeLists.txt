cmake_minimum_required(VERSION 3.24)

project(VulkanCompute)

set(PROJECT_SOURCES
        src/main.cpp
        src/vkcore.cpp
)
set(MAIN_TARGET VulkanCompute)

add_executable(${MAIN_TARGET})
target_sources(${MAIN_TARGET} PRIVATE ${PROJECT_SOURCES})

set_property(TARGET ${MAIN_TARGET} PROPERTY CXX_STANDARD 23)
set(VKFFT_BACKEND 0 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL, 4 - Level Zero, 5 - Metal")
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
target_compile_options(${MAIN_TARGET} PRIVATE -Wall -Wextra -Wpedantic)

add_definitions(-DVKFFT_BACKEND=${VKFFT_BACKEND})
target_compile_definitions(VulkanCompute PUBLIC -DVK_API_VERSION=13)
target_include_directories(${MAIN_TARGET} PRIVATE include external/eigen-5.0.1)
add_library(vkfft INTERFACE IMPORTED GLOBAL)
target_link_libraries(vkfft INTERFACE glslang::MachineIndependent glslang::OSDependent glslang::GenericCodeGen glslang::SPIRV)
set_target_properties(vkfft PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/external/vkFFT)

find_package(Vulkan REQUIRED COMPONENTS glslang SPIRV-Tools)
find_package(glslang REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)
# include(ExternalProject)
# set(EIGEN_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen-install/")
# ExternalProject_Add(
#   eigen
#   URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
#   SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen-src"
#   BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen-build"
#   INSTALL_DIR "${EIGEN_INSTALL_DIR}"
#   CMAKE_ARGS
#   -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
#   -DCMAKE_BUILD_TYPE=Release
# )
#
# file(MAKE_DIRECTORY ${EIGEN_INSTALL_DIR}/include)
# file(MAKE_DIRECTORY ${EIGEN_INSTALL_DIR}/include)
# add_library(eigenlib INTERFACE IMPORTED GLOBAL)
# add_dependencies(eigenlib eigen)
# target_compile_features(eigenlib INTERFACE cxx_std_23)
# set_target_properties(eigenlib PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${EIGEN_INSTALL_DIR}/include/eigen5)
# find_package(glslang REQUIRED)
# find_package(SDL3 REQUIRED CONFIG)
# find_package(VulkanMemoryAllocator CONFIG REQUIRED)
# add_library(VkFFT INTERFACE)
# target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/ /usr/include/glslang/Include)
# target_link_libraries(VkFFT INTERFACE glslang::MachineIndependent glslang::OSDependent glslang::GenericCodeGen glslang::SPIRV)
# target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=${VKFFT_BACKEND} -DVKFFT_MAX_FFT_DIMENSIONS=3)
if(result AND CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Using LTO")
  set_property(TARGET ${MAIN_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(result)
  message(STATUS "Build type is not Release, skipping LTO")
else()
  message(WARNING "LTO is not supported: ${output}")
endif()


target_precompile_headers(${MAIN_TARGET} PRIVATE include/hack.hpp)
target_link_libraries(${MAIN_TARGET} PRIVATE vkfft Vulkan::Vulkan hdf5::hdf5)
