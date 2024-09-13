#include "hack.hpp"
#include "vk_mem_alloc.h"
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>

const vk::MemoryBarrier fullMemoryBarrier(vk::AccessFlagBits::eShaderRead |
                                              vk::AccessFlagBits::eMemoryWrite,
                                          vk::AccessFlagBits::eMemoryRead |
                                              vk::AccessFlagBits::eMemoryWrite);

struct ComputeInfo {
  std::vector<vk::raii::Pipeline> pipeline;
  vk::raii::PipelineLayout layout;
  vk::raii::PipelineCache pipelineCache;
  vk::raii::DescriptorSet descriptorSet;
  uint32_t X;
  uint32_t Y;
  uint32_t Z;
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

struct RaiiVmaAllocator {
  VmaAllocator allocator;
  RaiiVmaAllocator(vk::raii::PhysicalDevice& physicalDevice,
                   vk::raii::Device& device, vk::raii::Instance& instance);
  ~RaiiVmaAllocator();
};

ComputeInfo setupPipelines(vk::raii::Device& device,
                           const std::vector<std::string>& binNames,
                           std::vector<vk::DescriptorType> bufferTypes,
                           std::vector<RaiiVmaBuffer*> buffers);
void appendPipeline(vk::raii::CommandBuffer& commandBuffer,
                    const ComputeInfo& cInfo, uint32_t n);

std::vector<uint32_t> readFile(const std::string& filename);
vk::raii::Instance makeInstance(const vk::raii::Context& context);
vk::raii::PhysicalDevice pickPhysicalDevice(const vk::raii::Instance& instance,
                                            const int32_t desiredGPU = -1);
ComputeInfo recordComputePipeline(vk::raii::CommandBuffer& commandBuffer,
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
