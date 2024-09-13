#include "vkhelpers.hpp"
#include "hack.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

std::vector<uint32_t> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> buffer(fileSize / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
  file.close();
  return buffer;
}

RaiiVmaAllocator::RaiiVmaAllocator(vk::raii::PhysicalDevice& physicalDevice,
                                   vk::raii::Device& device,
                                   vk::raii::Instance& instance) {
  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.physicalDevice = *physicalDevice;
  allocatorInfo.vulkanApiVersion = physicalDevice.getProperties().apiVersion;
  allocatorInfo.device = *device;
  allocatorInfo.instance = *instance;
  vmaCreateAllocator(&allocatorInfo, &allocator);
}

RaiiVmaAllocator::~RaiiVmaAllocator() {
  std::cout << "Destroying allocator\n";
  vmaDestroyAllocator(allocator);
}

void setupPipelines(vk::raii::Device& device,
                    const std::vector<std::string>& binNames,
                    std::vector<vk::DescriptorType> buffers) {
  std::vector<vk::raii::ShaderModule> modules;
  for (const auto& name : binNames) {
    auto shaderCode = readFile(name);
    vk::ShaderModuleCreateInfo shaderMCI(vk::ShaderModuleCreateFlags(),
                                         shaderCode);
    modules.emplace_back(vk::raii::ShaderModule(device, shaderMCI));
  }

  std::vector<vk::DescriptorSetLayoutBinding> dSLBs;
  for (uint32_t i = 0; i < buffers.size(); i++) {
    dSLBs.emplace_back(i, buffers[i], 1, vk::ShaderStageFlagBits::eCompute);
  }

  vk::DescriptorSetLayoutCreateInfo dSLCI(vk::DescriptorSetLayoutCreateFlags(),
                                          dSLBs);
  vk::raii::DescriptorSetLayout descriptorSetLayout(device, dSLCI);

  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(),
                                    *descriptorSetLayout);
  vk::raii::PipelineLayout pipelineLayout(device, pLCI);
  vk::raii::PipelineCache pipelineCache(device, vk::PipelineCacheCreateInfo());
  std::vector<vk::raii::Pipeline> pipelines;
  for (const auto& mod : modules) {
    vk::PipelineShaderStageCreateInfo cSCI(vk::PipelineShaderStageCreateFlags(),
                                           vk::ShaderStageFlagBits::eCompute,
                                           *mod, "main");
    vk::ComputePipelineCreateInfo cPCI(vk::PipelineCreateFlags(), cSCI,
                                       *pipelineLayout);
    pipelines.emplace_back(device, pipelineCache, cPCI);
  }
  vk::DescriptorPoolSize dPS(vk::DescriptorType::eStorageBuffer, 1);
  vk::DescriptorPoolCreateInfo dPCI(
      vk::DescriptorPoolCreateFlags(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
      1, dPS);
  vk::raii::DescriptorPool descriptorPool(device, dPCI);
  vk::DescriptorSetAllocateInfo dSAI(*descriptorPool, 1, &*descriptorSetLayout);
  vk::raii::DescriptorSets pDescriptorSets(device, dSAI);
  vk::raii::DescriptorSet descriptorSet(std::move(pDescriptorSets[0]));
  std::vector<vk::DescriptorBufferInfo> dBIs;
}

RaiiVmaBuffer::RaiiVmaBuffer(VmaAllocator& Allocator,
                             VmaAllocationCreateInfo& allocCreateInfo,
                             vk::BufferCreateInfo& BCI) {
  allocator = &Allocator;
  VkBuffer bufferRaw;
  vmaCreateBuffer(*allocator, reinterpret_cast<VkBufferCreateInfo*>(&BCI),
                  &allocCreateInfo, &bufferRaw, &allocation, &allocationInfo);
  buffer = bufferRaw;
}
RaiiVmaBuffer::~RaiiVmaBuffer() {
  std::cout << "Destroying buffer\n";
  vmaDestroyBuffer(*allocator, buffer, allocation);
}

void recordComputePipeline(vk::raii::CommandBuffer& commandBuffer,
                           ComputeInfo ci) {
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                fullMemoryBarrier, nullptr, nullptr);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *ci.pipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *ci.layout,
                                   0, {*ci.descriptorSet}, {});
  commandBuffer.dispatch(ci.X, ci.Y, ci.Z);
};

vk::raii::Instance makeInstance(const vk::raii::Context& context) {
  vk::ApplicationInfo appInfo{
      "VulkanCompute",   // Application Name
      1,                 // Application Version
      nullptr,           // Engine Name or nullptr
      0,                 // Engine Version
      VK_API_VERSION_1_3 // Vulkan API version
  };

  const std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
  vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(), // Flags
                                            &appInfo, // Application Info
                                            layers,   // Layers
                                            {});      // Extensions
  return vk::raii::Instance(context, instanceCreateInfo);
}

vk::raii::PhysicalDevice pickPhysicalDevice(const vk::raii::Instance& instance,
                                            const int32_t desiredGPU) {
  // check if there are GPUs that support Vulkan and "intelligently" select
  // one. Prioritises discrete GPUs, and after that VRAM size.
  vk::raii::PhysicalDevices physicalDevices(instance);
  uint32_t nDevices = physicalDevices.size();

  // shortcut if there's only one device available.
  if (nDevices == 1) {
    return vk::raii::PhysicalDevice(std::move(physicalDevices[0]));
  }
  // Try to select desired GPU if specified.
  if (desiredGPU > -1) {
    if (desiredGPU < static_cast<int32_t>(nDevices)) {
      return vk::raii::PhysicalDevice(std::move(physicalDevices[desiredGPU]));
    } else {
      std::cout << "Device not available\n";
    }
  }

  std::vector<uint32_t> discrete; // the indices of the available discrete gpus
  std::vector<uint64_t> vram(nDevices);
  for (uint32_t i = 0; i < nDevices; i++) {
    if (physicalDevices[i].getProperties().deviceType ==
        vk::PhysicalDeviceType::eDiscreteGpu) {
      discrete.push_back(i);
    }

    auto heaps = physicalDevices[i].getMemoryProperties().memoryHeaps;
    for (const auto& heap : heaps) {
      if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
        vram[i] = heap.size;
      }
    }
  }

  // only consider discrete gpus if available´:
  if (discrete.size() > 0) {
    if (discrete.size() == 1) {
      return vk::raii::PhysicalDevice(std::move(physicalDevices[discrete[0]]));
    } else {
      uint32_t max = 0;
      uint32_t selectedGPU = 0;
      for (const auto& index : discrete) {
        if (vram[index] > max) {
          max = vram[index];
          selectedGPU = index;
        }
      }
      return vk::raii::PhysicalDevice(std::move(physicalDevices[selectedGPU]));
    }
  } else {
    uint32_t max = 0;
    uint32_t selectedGPU = 0;
    for (uint32_t i = 0; i < nDevices; i++) {
      if (vram[i] > max) {
        max = vram[i];
        selectedGPU = i;
      }
    }
    return vk::raii::PhysicalDevice(std::move(physicalDevices[selectedGPU]));
  }
}
