#include "hack.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>

#include "vkFFT.h"
#include <complex>
#include <vulkan/vulkan_enums.hpp>

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
  RaiiVmaBuffer pump(allocator.allocator, allocCreateInfo, floatBCI);
  RaiiVmaBuffer oldpsir(allocator.allocator, allocCreateInfo, stateBCI);
  RaiiVmaBuffer kTimeEvo(allocator.allocator, allocCreateInfo, stateBCI);
  std::vector<RaiiVmaBuffer*> buffers{&psir, &psik,    &nR,
                                      &pump, &oldpsir, &kTimeEvo};

  c32* sStagingPtr = static_cast<c32*>(staging.allocationInfo.pMappedData);
  float* fStagingPtr = static_cast<float*>(staging.allocationInfo.pMappedData);
  for (uint32_t i = 0; i < nElementsY * nElementsX; i++) {
    fStagingPtr[i] = 0.;
  }
  oneTimeSubmit(
      device, commandPool, queue, [&](vk::CommandBuffer const& commandBuffer) {
        commandBuffer.copyBuffer(staging.buffer, nR.buffer,
                                 vk::BufferCopy(0, 0, reservoirBufferSize));
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

  std::vector<std::string> binNames{"rstep.spv", "kstep.spv", "finalstep.spv"};
  ComputeInfo computeInfo =
      setupPipelines(device, binNames,
                     std::vector({vk::DescriptorType::eStorageBuffer,
                                  vk::DescriptorType::eStorageBuffer,
                                  vk::DescriptorType::eStorageBuffer,
                                  vk::DescriptorType::eStorageBuffer,
                                  vk::DescriptorType::eStorageBuffer,
                                  vk::DescriptorType::eStorageBuffer}),
                     buffers);

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
  conf.isInputFormatted = true;
  conf.bufferNum = 1;
  conf.inputBuffer = (VkBuffer*)&psir.buffer;
  conf.buffer = (VkBuffer*)&psik.buffer;
  conf.bufferSize = &stateBufferSize;
  conf.inputBufferSize = &stateBufferSize;
  conf.inverseReturnToInputBuffer = true;

  auto resFFT = initializeVkFFT(&rToK, conf);
  VkFFTLaunchParams launchParams{};

  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  launchParams.commandBuffer = (VkCommandBuffer*)&*commandBuffer;
  appendPipeline(commandBuffer, computeInfo, 0);
  resFFT = VkFFTAppend(&rToK, -1, &launchParams);
  appendPipeline(commandBuffer, computeInfo, 1);
  resFFT = VkFFTAppend(&rToK, 1, &launchParams);
  appendPipeline(commandBuffer, computeInfo, 2);
  commandBuffer.end();

  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &*commandBuffer);
  for (uint32_t i = 0; i < nSteps; i++) {
    queue.submit(submitInfo, *fence);
    auto result = device.waitForFences({*fence}, true, -1);
  }

  oneTimeSubmit(
      device, commandPool, queue, [&](vk::CommandBuffer const& commandBuffer) {
        commandBuffer.copyBuffer(psir.buffer, staging.buffer,
                                 vk::BufferCopy(0, 0, stateBufferSize));
      });
  float* ePtr = reinterpret_cast<float*>(staging.allocationInfo.pMappedData);
  for (uint32_t j = 0; j < nElementsY; j++) {
    for (uint32_t i = 0; i < nElementsX; i++) {
      std::cout << ePtr[j * nElementsX + i] << ' ';
    }
    std::cout << '\n';
  }

  deleteVkFFT(&rToK);

  return 0;
}
