#include <numeric>
#define VMA_IMPLEMENTATION
#include "hack.hpp"
#include "vk_mem_alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>

#include "vkFFT.h"
#include <complex>

#include "vkhelpers.hpp"

typedef std::complex<double> c64;
typedef std::complex<float> c32;

template <typename T> constexpr T square(T x) { return x * x; }

constexpr float hbar = 6.582119569e-1;
constexpr float Pstrength = 24;
// constexpr uint32_t nSamples = 1024; ignore for now, just want to test if this
// gives reasonable results
// constexpr uint32_t sampleSpacing = 10;
// constexpr uint32_t warmupIters = 2000;
constexpr uint32_t nSteps = 10000;

/* good constants
alpha = 0.0004
gammalp = 0.2
Gamma = 0.1
G = 0.002
R = 0.015
eta = 2
pumpStrength = 24
sigma = 1.2
dt = 0.2
samplesX = 512
samplesY = 512
startX = -140
endX = 140
startY = -140
endY = 140
m = 0.32
nframes = 512
prerun = 4000
intermediaterun = 1000
*/

struct SimConstants {
  uint32_t nElementsX;
  uint32_t nElementsY;
  float alpha;
  float gammalp;
  float Gamma;
  float m;
  float G;
  float R;
  float eta;
  float pumpStrength;
  float L0;
  float r0;
  float beta;
  float startX;
  float endX;
  float startY;
  float endY;
  float dt;
};

int main(int argc, char* argv[]) {
  vk::raii::Context context;
  auto instance = makeInstance(context);
  auto physicalDevice = pickPhysicalDevice(instance);
  auto queueFamilyProps = physicalDevice.getQueueFamilyProperties();
  auto propIt =
      std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(),
                   [](const vk::QueueFamilyProperties& prop) {
                     return prop.queueFlags & vk::QueueFlagBits::eCompute;
                   });
  const uint32_t computeQueueFamilyIndex =
      std::distance(queueFamilyProps.begin(), propIt);

  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo dQCI(vk::DeviceQueueCreateFlags(),
                                 computeQueueFamilyIndex, 1, &queuePriority);
  vk::DeviceCreateInfo dCI(vk::DeviceCreateFlags(), dQCI);
  vk::raii::Device device(physicalDevice, dCI);
  vk::raii::Queue queue(device, computeQueueFamilyIndex, 0);
  vk::raii::Fence fence(device, vk::FenceCreateInfo());

  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  computeQueueFamilyIndex);
  vk::raii::CommandPool commandPool(device, commandPoolCreateInfo);
  vk::CommandBufferAllocateInfo cBAI(*commandPool,
                                     vk::CommandBufferLevel::ePrimary, 1);
  vk::raii::CommandBuffers commandBuffers(device, cBAI);
  vk::raii::CommandBuffer commandBuffer(std::move(commandBuffers[0]));

  constexpr uint64_t nElementsX = 512;
  constexpr uint64_t nElementsY = 512;
  constexpr float dt = 0.1;
  uint64_t stateBufferSize = nElementsX * nElementsY * sizeof(c32);
  uint64_t reservoirBufferSize = nElementsX * nElementsY * sizeof(float);

  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.physicalDevice = *physicalDevice;
  allocatorInfo.vulkanApiVersion = physicalDevice.getProperties().apiVersion;
  allocatorInfo.device = *device;
  allocatorInfo.instance = *instance;

  RaiiVmaAllocator allocator(physicalDevice, device, instance);

  vk::BufferCreateInfo stagingBCI({}, stateBufferSize,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst);

  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  RaiiVmaBuffer staging(allocator.allocator, allocCreateInfo, stagingBCI);

  vk::BufferCreateInfo stateBCI{vk::BufferCreateFlags(),
                                stateBufferSize,
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst |
                                    vk::BufferUsageFlagBits::eTransferSrc,
                                vk::SharingMode::eExclusive,
                                1,
                                &computeQueueFamilyIndex};
  vk::BufferCreateInfo floatBCI{vk::BufferCreateFlags(),
                                reservoirBufferSize,
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst |
                                    vk::BufferUsageFlagBits::eTransferSrc,
                                vk::SharingMode::eExclusive,
                                1,
                                &computeQueueFamilyIndex};
  allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocCreateInfo.priority = 1.0f;
  RaiiVmaBuffer psir(allocator.allocator, allocCreateInfo, stateBCI);
  RaiiVmaBuffer psik(allocator.allocator, allocCreateInfo, stateBCI);
  RaiiVmaBuffer nR(allocator.allocator, allocCreateInfo, floatBCI);
  RaiiVmaBuffer kTimeEvo(allocator.allocator, allocCreateInfo, stateBCI);
  RaiiVmaBuffer pump(allocator.allocator, allocCreateInfo, floatBCI);
  c32* sStagingPtr = static_cast<c32*>(staging.allocationInfo.pMappedData);
  float* fStagingPtr = static_cast<float*>(staging.allocationInfo.pMappedData);
  for (uint32_t i = 0; i < nElementsY * nElementsX; i++) {
    fStagingPtr[i] = 0.;
  }
  oneTimeSubmit(
      device, commandPool, queue, [&](vk::CommandBuffer const& commandBuffer) {
        commandBuffer.copyBuffer(staging.buffer, psir.buffer,
                                 vk::BufferCopy(0, 0, stateBufferSize));
      });

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-0.1, 0.1);
  for (int32_t j = 0; j < nElementsY; j++) {
    for (int32_t i = 0; i < nElementsX; i++) {
      sStagingPtr[j * nElementsX + i] = c32{dis(gen), dis(gen)};
    }
  }
  oneTimeSubmit(
      device, commandPool, queue, [&](vk::CommandBuffer const& commandBuffer) {
        commandBuffer.copyBuffer(staging.buffer, psir.buffer,
                                 vk::BufferCopy(0, 0, stateBufferSize));
      });

  auto shaderCode = readFile("step.spv");
  vk::ShaderModuleCreateInfo shaderMCI(vk::ShaderModuleCreateFlags(),
                                       shaderCode);
  vk::raii::ShaderModule shaderModule(device, shaderMCI);

  const std::vector<vk::DescriptorSetLayoutBinding> dSLBs = {
      {0, vk::DescriptorType::eUniformBuffer, 1,
       vk::ShaderStageFlagBits::eCompute},
      {1, vk::DescriptorType::eStorageBuffer, 1,
       vk::ShaderStageFlagBits::eCompute},
      {2, vk::DescriptorType::eStorageBuffer, 1,
       vk::ShaderStageFlagBits::eCompute},
      {3, vk::DescriptorType::eStorageBuffer, 1,
       vk::ShaderStageFlagBits::eCompute}};

  vk::DescriptorSetLayoutCreateInfo dSLCI(vk::DescriptorSetLayoutCreateFlags(),
                                          dSLBs);
  vk::raii::DescriptorSetLayout descriptorSetLayout(device, dSLCI);

  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(),
                                    *descriptorSetLayout);
  vk::raii::PipelineLayout pipelineLayout(device, pLCI);
  vk::raii::PipelineCache pipelineCache(device, vk::PipelineCacheCreateInfo());
  vk::PipelineShaderStageCreateInfo rstepSCI(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
      *shaderModule, "rstep");
  vk::PipelineShaderStageCreateInfo kstepSCI(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
      *shaderModule, "kstep");
  vk::PipelineShaderStageCreateInfo nstepSCI(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
      *shaderModule, "nstep");
  vk::ComputePipelineCreateInfo rstepPCI(vk::PipelineCreateFlags(), rstepSCI,
                                         *pipelineLayout);
  vk::ComputePipelineCreateInfo kstepPCI(vk::PipelineCreateFlags(), kstepSCI,
                                         *pipelineLayout);
  vk::ComputePipelineCreateInfo nstepPCI(vk::PipelineCreateFlags(), nstepSCI,
                                         *pipelineLayout);
  vk::raii::Pipeline rstepPipeline(device, pipelineCache, rstepPCI);
  vk::raii::Pipeline kstepPipeline(device, pipelineCache, kstepPCI);
  vk::raii::Pipeline nstepPipeline(device, pipelineCache, nstepPCI);

  vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                                            1);
  vk::DescriptorPoolCreateInfo dPCI(
      vk::DescriptorPoolCreateFlags(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
      1, descriptorPoolSize);
  vk::raii::DescriptorPool descriptorPool(device, dPCI);

  vk::DescriptorSetAllocateInfo dSAI(*descriptorPool, 1, &*descriptorSetLayout);
  vk::raii::DescriptorSets pDescriptorSets(device, dSAI);
  vk::raii::DescriptorSet descriptorSet(std::move(pDescriptorSets[0]));
  vk::DescriptorBufferInfo rspaceBufferInfo(rspaceBuffer, 0, stateBufferSize);
  vk::DescriptorBufferInfo kspaceBufferInfo(kspaceBuffer, 0, stateBufferSize);
  vk::DescriptorBufferInfo reservoirBufferInfo(reservoirBuffer, 0,
                                               reservoirBufferSize);

  const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
      {*descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr,
       &rspaceBufferInfo},
      {*descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &rspaceBufferInfo},
      {*descriptorSet, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &kspaceBufferInfo},
      {*descriptorSet, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &reservoirBufferInfo}};
  device.updateDescriptorSets(writeDescriptorSets, {});

  VkFFTConfiguration conf{};
  VkFFTApplication rToK{};
  conf.device = (VkDevice*)&*device;
  conf.FFTdim = 2;
  conf.size[0] = nElementsX;
  conf.size[1] = nElementsY;
  conf.numberBatches = 1;
  conf.queue = (VkQueue*)&*queue;
  conf.fence = (VkFence*)&*fence;
  conf.commandPool = (VkCommandPool*)&*commandPool;
  conf.physicalDevice = (VkPhysicalDevice*)&*physicalDevice;
  conf.bufferNum = 1;
  conf.inputBuffer = reinterpret_cast<VkBuffer*>(&rspaceBuffer);
  conf.outputBuffer = reinterpret_cast<VkBuffer*>(&kspaceBuffer);
  conf.bufferSize = &stateBufferSize;

  auto resFFT = initializeVkFFT(&rToK, conf);
  VkFFTApplication kToR{};
  conf.device = (VkDevice*)&*device;
  conf.FFTdim = 2;
  conf.size[0] = nElementsX;
  conf.size[1] = nElementsY;
  conf.numberBatches = 1;
  conf.queue = (VkQueue*)&*queue;
  conf.fence = (VkFence*)&*fence;
  conf.commandPool = (VkCommandPool*)&*commandPool;
  conf.physicalDevice = (VkPhysicalDevice*)&*physicalDevice;
  conf.bufferNum = 1;
  conf.inputBuffer = reinterpret_cast<VkBuffer*>(&kspaceBuffer);
  conf.outputBuffer = reinterpret_cast<VkBuffer*>(&rspaceBuffer);
  conf.bufferSize = &stateBufferSize;

  resFFT = initializeVkFFT(&kToR, conf);
  VkFFTLaunchParams launchParams{};

  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  launchParams.commandBuffer = (VkCommandBuffer*)&*commandBuffer;
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *rstepPipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   *pipelineLayout, 0, {*descriptorSet}, {});
  commandBuffer.dispatch(nElementsX, nElementsY, 1);
  vk::MemoryBarrier memoryBarrier(
      vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eMemoryWrite,
      vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                memoryBarrier, nullptr, nullptr);
  resFFT = VkFFTAppend(&rToK, -1, &launchParams);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                memoryBarrier, nullptr, nullptr);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *kstepPipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   *pipelineLayout, 0, {*descriptorSet}, {});
  commandBuffer.dispatch(nElementsX, nElementsY, 1);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                memoryBarrier, nullptr, nullptr);
  resFFT = VkFFTAppend(&kToR, 1, &launchParams);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                memoryBarrier, nullptr, nullptr);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *nstepPipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   *pipelineLayout, 0, {*descriptorSet}, {});
  commandBuffer.end();

  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &*commandBuffer);
  for (uint32_t i = 0; i < nSteps; i++) {
    queue.submit(submitInfo, *fence);
    auto result = device.waitForFences({*fence}, true, -1);
  }

  oneTimeSubmit(
      device, commandPool, queue, [&](vk::CommandBuffer const& commandBuffer) {
        commandBuffer.copyBuffer(rspaceBuffer, stagingBuffer,
                                 vk::BufferCopy(0, 0, stateBufferSize));
      });
  float* ePtr = reinterpret_cast<float*>(stAI.pMappedData);
  for (uint32_t j = 0; j < nElementsY; j++) {
    for (uint32_t i = 0; i < nElementsX; i++) {
      std::cout << ePtr[j * nElementsX + i] << ' ';
    }
    std::cout << '\n';
  }

  deleteVkFFT(&rToK);
  deleteVkFFT(&kToR);
  vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);
  vmaDestroyBuffer(allocator, rspaceBuffer, rspaceAlloc);
  vmaDestroyBuffer(allocator, kspaceBuffer, kspaceAlloc);
  vmaDestroyBuffer(allocator, reservoirBuffer, reservoirAlloc);
  vmaDestroyAllocator(allocator);

  return 0;
}
