#define VMA_IMPLEMENTATION
#include "hack.hpp"
#include "vk_mem_alloc.h"

const vk::MemoryBarrier fullMemoryBarrier(vk::AccessFlagBits::eShaderRead |
                                              vk::AccessFlagBits::eMemoryWrite,
                                          vk::AccessFlagBits::eMemoryRead |
                                              vk::AccessFlagBits::eMemoryWrite);

struct ComputeInfo {
  vk::raii::Pipeline pipeline;
  vk::raii::PipelineLayout layout;
  vk::raii::DescriptorSet descriptorSet;
  uint32_t X;
  uint32_t Y;
  uint32_t Z;
};

struct RaiiVmaAllocator {
  VmaAllocator allocator;
  RaiiVmaAllocator(vk::raii::PhysicalDevice& physicalDevice,
                   vk::raii::Device& device, vk::raii::Instance& instance);
  ~RaiiVmaAllocator();
};

struct RaiiVmaBuffer {
  // RaiiVmaAllocator takes care of deleting itself
  VmaAllocator* allocator;
  vk::Buffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo allocationInfo;
  RaiiVmaBuffer(VmaAllocator& Allocator,
                VmaAllocationCreateInfo& allocCreateInfo,
                vk::BufferCreateInfo& BCI);
  ~RaiiVmaBuffer();
};

std::vector<uint32_t> readFile(const std::string& filename);
vk::raii::Instance makeInstance(const vk::raii::Context& context);
vk::raii::PhysicalDevice pickPhysicalDevice(const vk::raii::Instance& instance,
                                            const int32_t desiredGPU = -1);
void recordComputePipeline(vk::raii::CommandBuffer& commandBuffer,
                           ComputeInfo ci);

template <typename Func>
void oneTimeSubmit(const vk::raii::Device& device,
                   const vk::raii::CommandPool& commandPool,
                   const vk::raii::Queue& queue, const Func& func) {
  vk::raii::CommandBuffer commandBuffer =
      std::move(vk::raii::CommandBuffers(
                    device, {*commandPool, vk::CommandBufferLevel::ePrimary, 1})
                    .front());
  commandBuffer.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  func(*commandBuffer);
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, *commandBuffer);
  queue.submit(submitInfo, nullptr);
  queue.waitIdle();
}
