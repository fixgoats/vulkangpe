cmake_minimum_required(VERSION 3.24)

project(VulkanCompute)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_C_STANDARD 99)


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
target_include_directories(${MAIN_TARGET} PRIVATE include)
add_subdirectory(vk-bootstrap)

find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)
find_package(SDL3 REQUIRED CONFIG)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/ /usr/include/glslang/Include)
target_link_libraries(VkFFT INTERFACE glslang::MachineIndependent glslang::OSDependent glslang::GenericCodeGen glslang::SPIRV)
target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=${VKFFT_BACKEND} -DVKFFT_MAX_FFT_DIMENSIONS=3)
if(result AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Using LTO")
    set_property(TARGET ${MAIN_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "LTO is not supported: ${output}")
endif()


install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/" DESTINATION include)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
target_precompile_headers(${MAIN_TARGET} PRIVATE include/hack.hpp)
target_link_libraries(${MAIN_TARGET} PRIVATE SPIRV glslang Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator VkFFT SDL3::SDL3)
