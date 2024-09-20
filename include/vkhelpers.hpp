#pragma once
#include "hack.hpp"
#include "vk_mem_alloc.h"
#include <complex>
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>

typedef std::complex<double> c64;
typedef std::complex<float> c32;

struct SimConstants {
  uint32_t nElementsX;
  uint32_t nElementsY;
  uint32_t xGroupSize;
  uint32_t yGroupSize;
  float alpha;
  float gammalp;
  float Gamma;
  float G;
  float R;
  float eta;
  float dt;
  constexpr uint32_t X() const { return nElementsX / xGroupSize; }
  constexpr uint32_t Y() const { return nElementsY / yGroupSize; }
  constexpr bool validate() const {
    return (nElementsY % yGroupSize == 0) && (nElementsX % xGroupSize == 0);
  }
  constexpr uint32_t elementsTotal() const { return nElementsX * nElementsY; }
};

const vk::MemoryBarrier fullMemoryBarrier(vk::AccessFlagBits::eShaderRead |
                                              vk::AccessFlagBits::eMemoryWrite,
                                          vk::AccessFlagBits::eMemoryRead |
                                              vk::AccessFlagBits::eMemoryWrite);

struct MetaBuffer {
  vk::Buffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo aInfo;
  MetaBuffer();
  MetaBuffer(VmaAllocator& allocator, VmaAllocationCreateInfo& allocCreateInfo,
             vk::BufferCreateInfo& BCI);
  // To call on default constructed metabuffer
  void allocate(VmaAllocator& allocator,
                VmaAllocationCreateInfo& allocCreateInfo,
                vk::BufferCreateInfo& BCI);
  void extirpate(VmaAllocator& allocator);
};

std::vector<uint32_t> readFile(const std::string& filename);
vk::raii::Instance makeInstance(const vk::raii::Context& context);
vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance,
                                      const int32_t desiredGPU = -1);

template <typename Func>
void oneTimeSubmit(const vk::Device& device, const vk::CommandPool& commandPool,
                   const vk::Queue& queue, const Func& func) {
  vk::CommandBuffer commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  commandBuffer.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  func(commandBuffer);
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  queue.submit(submitInfo, nullptr);
  queue.waitIdle();
}
