#include "hack.hpp"

std::vector<uint32_t> readFile(const std::string& filename);
vk::raii::Instance makeInstance(const vk::raii::Context& context);
vk::raii::PhysicalDevice pickPhysicalDevice(const vk::raii::Instance& instance,
                                            const int32_t desiredGPU = -1);
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
