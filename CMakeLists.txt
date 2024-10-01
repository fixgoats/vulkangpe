cmake_minimum_required(VERSION 3.24)

project(VulkanCompute)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 REQUIRED COMPONENTS Widgets)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
        src/main.cpp
        src/vkhelpers.cpp
        src/mainwindow.cpp
        src/mainwindow.ui
        src/GPEsim.cpp
        include/mainwindow.h
)

qt_add_executable(VulkanCompute
    MANUAL_FINALIZATION
    ${PROJECT_SOURCES}
)

target_include_directories(VulkanCompute PRIVATE include)
set_property(TARGET VulkanCompute PROPERTY CXX_STANDARD 20)
set(VKFFT_BACKEND 0 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL, 4 - Level Zero, 5 - Metal")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

add_definitions(-DVKFFT_BACKEND=${VKFFT_BACKEND})
target_compile_definitions(VulkanCompute PUBLIC -DVK_API_VERSION=13)

find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/ /usr/include/glslang/Include)
target_link_libraries(VkFFT INTERFACE glslang::MachineIndependent glslang::OSDependent glslang::GenericCodeGen glslang::SPIRV)
target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=${VKFFT_BACKEND} -DVKFFT_MAX_FFT_DIMENSIONS=3)
if(result)
    message(STATUS "Using LTO")
    set_property(TARGET VulkanCompute PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "LTO is not supported: ${output}")
endif()


install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/" DESTINATION include)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
target_precompile_headers(VulkanCompute PRIVATE include/hack.hpp)
target_link_libraries(VulkanCompute PRIVATE SPIRV glslang Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator VkFFT Qt6::Widgets ${OpenCV_LIBS})
qt_finalize_executable(VulkanCompute)
