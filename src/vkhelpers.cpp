#include <cstddef>
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "vkhelpers.hpp"
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
/*
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

RaiiVmaAllocator::~RaiiVmaAllocator() { vmaDestroyAllocator(allocator); }

ComputeInfo setupPipelines(vk::raii::Device& device,
                           const std::vector<std::string>& binNames,
                           std::vector<vk::DescriptorType> bufferTypes,
                           std::vector<RaiiVmaBuffer*> buffers,
                           SimConstants ahhh, uint32_t n) {
  std::vector<vk::raii::ShaderModule> modules;
  for (const auto& name : binNames) {
    auto shaderCode = readFile(name);
    vk::ShaderModuleCreateInfo shaderMCI(vk::ShaderModuleCreateFlags(),
                                         shaderCode);
    modules.emplace_back(vk::raii::ShaderModule(device, shaderMCI));
  }

  std::vector<vk::DescriptorSetLayoutBinding> dSLBs;
  for (uint32_t i = 0; i < bufferTypes.size(); i++) {
    dSLBs.emplace_back(i, bufferTypes[i], 1, vk::ShaderStageFlagBits::eCompute);
  }

  vk::DescriptorSetLayoutCreateInfo dSLCI(vk::DescriptorSetLayoutCreateFlags(),
                                          dSLBs);
  vk::raii::DescriptorSetLayout descriptorSetLayout(device, dSLCI);

  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(),
                                    *descriptorSetLayout);
  vk::raii::PipelineLayout pipelineLayout(device, pLCI);
  vk::raii::PipelineCache pipelineCache(device, vk::PipelineCacheCreateInfo());
  std::vector<vk::raii::Pipeline> pipelines;
  std::vector<vk::SpecializationMapEntry> bleh(n);
  for (uint32_t i = 0; i < n; i++) {
    bleh[i].constantID = i;
    bleh[i].offset = i * 4;
    bleh[i].size = 4;
  }
  vk::SpecializationInfo specInfo;
  specInfo.mapEntryCount = n;
  specInfo.pMapEntries = bleh.data();
  specInfo.dataSize = sizeof(SimConstants);
  specInfo.pData = &ahhh;
  assert(ahhh.nElementsX % ahhh.xGroupSize == 0);
  assert(ahhh.nElementsY % ahhh.yGroupSize == 0);
  uint32_t X = ahhh.nElementsX / ahhh.xGroupSize;
  uint32_t Y = ahhh.nElementsY / ahhh.yGroupSize;

  for (const auto& mod : modules) {
    vk::PipelineShaderStageCreateInfo cSCI(vk::PipelineShaderStageCreateFlags(),
                                           vk::ShaderStageFlagBits::eCompute,
                                           *mod, "main", &specInfo);
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
  for (const auto& b : buffers) {
    dBIs.emplace_back((*b).buffer, 0, (*b).allocationInfo.size);
  }
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  for (uint32_t i = 0; i < dBIs.size(); i++) {
    writeDescriptorSets.emplace_back(*descriptorSet, i, 0, 1, bufferTypes[i],
                                     nullptr, &dBIs[i]);
  }
  device.updateDescriptorSets(writeDescriptorSets, {});
  return ComputeInfo{std::move(pipelines),
                     std::move(pipelineLayout),
                     std::move(pipelineCache),
                     std::move(descriptorPool),
                     std::move(pDescriptorSets),
                     std::move(descriptorSet),
                     X,
                     Y,
                     1};
}

void appendPipeline(vk::raii::CommandBuffer& commandBuffer,
                    const ComputeInfo& cInfo, uint32_t n) {
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                fullMemoryBarrier, nullptr, nullptr);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                             *cInfo.pipeline[n]);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   cInfo.layout, 0, {*cInfo.descriptorSet}, {});
  commandBuffer.dispatch(cInfo.X, cInfo.Y, cInfo.Z);
}

void appendPipeline(vk::CommandBuffer& commandBuffer,
                    const vk::Pipeline& pipeline,
                    const vk::PipelineLayout& pipelineLayout,
                    const vk::DescriptorSet descriptorSet, WorkGroups groups) {
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                fullMemoryBarrier, nullptr, nullptr);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   pipelineLayout, 0, {descriptorSet}, {});
  commandBuffer.dispatch(groups.X, groups.Y, groups.Z);
}
*/

MetaBuffer::MetaBuffer() {
  buffer = vk::Buffer{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
}

MetaBuffer::MetaBuffer(VmaAllocator& allocator,
                       VmaAllocationCreateInfo& allocCreateInfo,
                       vk::BufferCreateInfo& BCI) {
  buffer = vk::Buffer{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
  vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&BCI),
                  &allocCreateInfo, reinterpret_cast<VkBuffer*>(&buffer),
                  &allocation, &aInfo);
}

void MetaBuffer::allocate(VmaAllocator& allocator,
                          VmaAllocationCreateInfo& allocCreateInfo,
                          vk::BufferCreateInfo& BCI) {
  vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&BCI),
                  &allocCreateInfo, reinterpret_cast<VkBuffer*>(&buffer),
                  &allocation, &aInfo);
}

void MetaBuffer::extirpate(VmaAllocator& allocator) {
  vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer), allocation);
}

/*vk::raii::Instance makeInstance(const vk::raii::Context& context) {
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
}*/

vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance,
                                      const int32_t desiredGPU) {
  // check if there are GPUs that support Vulkan and "intelligently" select
  // one. Prioritises discrete GPUs, and after that VRAM size.
  std::vector<vk::PhysicalDevice> pDevices =
      instance.enumeratePhysicalDevices();
  uint32_t nDevices = pDevices.size();

  // shortcut if there's only one device available.
  if (nDevices == 1) {
    return pDevices[0];
  }
  // Try to select desired GPU if specified.
  if (desiredGPU > -1) {
    if (desiredGPU < static_cast<int32_t>(nDevices)) {
      return pDevices[desiredGPU];
    } else {
      std::cout << "Device not available\n";
    }
  }

  std::vector<uint32_t> discrete; // the indices of the available discrete gpus
  std::vector<uint64_t> vram(nDevices);
  for (uint32_t i = 0; i < nDevices; i++) {
    if (pDevices[i].getProperties().deviceType ==
        vk::PhysicalDeviceType::eDiscreteGpu) {
      discrete.push_back(i);
    }

    auto heaps = pDevices[i].getMemoryProperties().memoryHeaps;
    for (const auto& heap : heaps) {
      if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
        vram[i] = heap.size;
      }
    }
  }

  // only consider discrete gpus if available´:
  if (discrete.size() > 0) {
    if (discrete.size() == 1) {
      return pDevices[discrete[0]];
    } else {
      uint32_t max = 0;
      uint32_t selectedGPU = 0;
      for (const auto& index : discrete) {
        if (vram[index] > max) {
          max = vram[index];
          selectedGPU = index;
        }
      }
      return pDevices[selectedGPU];
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
    return pDevices[selectedGPU];
  }
}
