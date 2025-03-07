#include "mathhelpers.h"
#include <cmath>
#include <cstddef>
#include <format>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>
#define VMA_IMPLEMENTATION 1003000
#include "SDL3/SDL_vulkan.h"
#include "vk_mem_alloc.h"
#include "vkcore.h"
#include <cstdint>
#include <iostream>
#include <set>

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

AllocatedImage::AllocatedImage() {
  img = vk::Image{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
}

AllocatedImage::AllocatedImage(VmaAllocator& allocator,
                               VmaAllocationCreateInfo& allocCreateInfo,
                               vk::ImageCreateInfo& BCI) {
  img = vk::Image{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
  vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&BCI),
                 &allocCreateInfo, reinterpret_cast<VkImage*>(&img),
                 &allocation, &aInfo);
}

void AllocatedImage::allocate(VmaAllocator& allocator,
                              VmaAllocationCreateInfo& allocCreateInfo,
                              vk::ImageCreateInfo& BCI) {
  vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&BCI),
                 &allocCreateInfo, reinterpret_cast<VkImage*>(&img),
                 &allocation, &aInfo);
}

Algorithm::Algorithm(vk::Device* device,
                     const std::vector<vk::ImageView>& img_views,
                     const std::vector<MetaBuffer*>& buffers,
                     const std::vector<u32>& spirv, const u8* specConsts,
                     const u32* sizes, size_t nConsts, const u32* pushSizes,
                     size_t nPushConstants) {
  p_device = device;
  vk::ShaderModuleCreateInfo shaderMCI(vk::ShaderModuleCreateFlags(), spirv);
  m_ShaderModule = device->createShaderModule(shaderMCI);
  std::vector<vk::DescriptorSetLayoutBinding> dSLBs(img_views.size() +
                                                    buffers.size());
  for (u32 i = 0; i < img_views.size(); i++) {
    dSLBs[i] = {i, vk::DescriptorType::eStorageImage, 1,
                vk::ShaderStageFlagBits::eCompute};
  }
  for (u32 i = img_views.size(); i < buffers.size() + img_views.size(); i++) {
    dSLBs[i] = {i, vk::DescriptorType::eStorageBuffer, 1,
                vk::ShaderStageFlagBits::eCompute};
  }
  vk::DescriptorSetLayoutCreateInfo dSLCI(vk::DescriptorSetLayoutCreateFlags(),
                                          dSLBs);
  m_DSL = device->createDescriptorSetLayout(dSLCI);
  std::vector<vk::PushConstantRange> ranges(nPushConstants);
  u32 pushOffsets = 0;
  for (u32 i = 0; i < nPushConstants; i++) {
    ranges[i] = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute,
                                      pushOffsets, pushSizes[i]);
    pushOffsets += pushSizes[i];
  }
  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(), m_DSL,
                                    ranges);
  m_PipelineLayout = device->createPipelineLayout(pLCI);
  std::vector<vk::SpecializationMapEntry> specEntries(nConsts);
  // specConsts needs to have no gaps. This can be done with field ordering or,
  // if the field order is important, with a packing directive. Packing was
  // quite inefficient for a long time since it can force unaligned accesses,
  // but modern systems are quite good at unaligned accesses, and this probably
  // won't be a hot path either way.
  u32 offset = 0;
  for (u32 i = 0; i < specEntries.size(); i++) {
    specEntries[i].constantID = i;
    specEntries[i].offset = offset;
    specEntries[i].size = sizes[i];
    offset += sizes[i];
  }
  vk::SpecializationInfo specInfo;
  specInfo.mapEntryCount = nConsts;
  specInfo.pMapEntries = specEntries.data();
  specInfo.dataSize = offset;
  specInfo.pData = specConsts;

  vk::PipelineShaderStageCreateInfo cSCI(vk::PipelineShaderStageCreateFlags(),
                                         vk::ShaderStageFlagBits::eCompute,
                                         m_ShaderModule, "main", &specInfo);
  vk::ComputePipelineCreateInfo cPCI(vk::PipelineCreateFlags(), cSCI,
                                     m_PipelineLayout);
  auto result = device->createComputePipeline({}, cPCI);
  m_Pipeline = result.value;

  // This is probably not the most efficient way to do this, but I'm not going
  // to mess around with the descriptors after creation so the only overhead
  // should be memory, and I'm not going to make thousands of these so
  // it should be fine.
  vk::DescriptorPoolSize dPS(vk::DescriptorType::eStorageBuffer, 1);
  vk::DescriptorPoolCreateInfo dPCI(
      vk::DescriptorPoolCreateFlags(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
      1, dPS);
  m_DescriptorPool = device->createDescriptorPool(dPCI);
  vk::DescriptorSetAllocateInfo dSAI(m_DescriptorPool, 1, &m_DSL);
  auto descriptorSets = device->allocateDescriptorSets(dSAI);
  m_DescriptorSet = descriptorSets[0];
  std::vector<vk::DescriptorImageInfo> dIIs(img_views.size());
  std::vector<vk::DescriptorBufferInfo> dBIs(buffers.size());
  for (size_t i = 0; i < dIIs.size(); i++) {
    dIIs[i] = {{}, img_views[i], vk::ImageLayout::eGeneral};
  }
  for (size_t i = 0; i < dBIs.size(); i++) {
    dBIs[i] =
        vk::DescriptorBufferInfo(buffers[i]->buffer, 0, buffers[i]->aInfo.size);
  }
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets(img_views.size() +
                                                          dBIs.size());
  for (u32 i = 0; i < dIIs.size(); i++) {
    writeDescriptorSets[i] = {m_DescriptorSet, i, 0,
                              vk::DescriptorType::eStorageImage, dIIs[i]};
  }
  for (uint32_t i = dIIs.size(); i < dIIs.size() + dBIs.size(); i++) {
    writeDescriptorSets[i] = {
        m_DescriptorSet, i,       0, 1, vk::DescriptorType::eStorageBuffer,
        nullptr,         &dBIs[i]};
  }
  device->updateDescriptorSets(writeDescriptorSets, {});
}

Algorithm::~Algorithm() {
  p_device->destroyDescriptorSetLayout(m_DSL);
  p_device->destroyDescriptorPool(m_DescriptorPool);
  p_device->destroyShaderModule(m_ShaderModule);
  p_device->destroyPipeline(m_Pipeline);
  p_device->destroyPipelineLayout(m_PipelineLayout);
}

/*std::ostream& operator<<(std::ostream& os, const SimConstants& obj) {
  os << std::format("(nX: {}, : nY{}, nZ: {}, xGS: {}, yGS: {}, gamma: {}, "
                    "Gamma: {}, R: {}, EXY: {}, dt: {}, B0: {}, width: {}, "
                    "resgamma: {})",
                    obj.nElementsX, obj.nElementsY, obj.nElementsZ,
                    obj.xGroupSize, obj.yGroupSize, obj.gamma, obj.Gamma, obj.R,
                    obj.EXY, obj.dt, obj.B0, obj.width, obj.resgamma);
  return os;
}

std::ofstream& operator<<(std::ofstream& os, const SimConstants& obj) {
  os << std::format("(nX: {}, : nY{}, nZ: {}, xGS: {}, yGS: {}, gamma: {}, "
                    "Gamma: {}, R: {}, EXY: {}, dt: {}, B0: {}, width: {}, "
                    "resgamma: {})",
                    obj.nElementsX, obj.nElementsY, obj.nElementsZ,
                    obj.xGroupSize, obj.yGroupSize, obj.gamma, obj.Gamma, obj.R,
                    obj.EXY, obj.dt, obj.B0, obj.width, obj.resgamma);
  return os;
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
    if (pDevices[0].getProperties().deviceType ==
            vk::PhysicalDeviceType::eIntegratedGpu or
        pDevices[0].getProperties().deviceType ==
            vk::PhysicalDeviceType::eCpu) {
      std::cout << "Only integrated GPU or CPU detected, you may not see much "
                   "benefit from GPU acceleration.\n";
    }
    return pDevices[0];
  }
  // Try to select desired GPU if specified.
  if (desiredGPU > -1) {
    if (desiredGPU < static_cast<int32_t>(nDevices)) {
      return pDevices[desiredGPU];
    } else {
      std::cout << "Selected device is not available.\n";
    }
  }

  std::vector<uint32_t> discrete; // the indices of the available discrete gpus
  std::vector<uint64_t> vram(nDevices);
  for (uint32_t i = 0; i < nDevices; i++) {
    if (pDevices[i].getProperties().deviceType ==
        vk::PhysicalDeviceType::eDiscreteGpu) {
      discrete.push_back(i);
    }

    // Gather reported VRAM sizes as an index to rank GPUs by.
    auto heaps = pDevices[i].getMemoryProperties().memoryHeaps;
    for (const auto& heap : heaps) {
      if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
        vram[i] = heap.size;
      }
    }
  }

  // only consider discrete gpus if available:
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

std::set<std::string> get_supported_extensions() {
  vk::Result result;
  uint32_t count = 0;
  result = vk::enumerateInstanceExtensionProperties(nullptr, &count, nullptr);
  if (result != vk::Result::eSuccess) {
    runtime_exc("Couldn't enumerate instance extension properties.");
  }

  std::vector<vk::ExtensionProperties> extensionProperties(count);

  // Get the extensions
  result = vk::enumerateInstanceExtensionProperties(nullptr, &count,
                                                    extensionProperties.data());
  if (result != vk::Result::eSuccess) {
    runtime_exc("Couldn't write instance extension properties to buffer");
  }

  std::set<std::string> extensions;
  for (auto& extension : extensionProperties) {
    extensions.insert(extension.extensionName);
  }

  return extensions;
}

static const std::string appName{"Vulkan GPE Simulator"};
static const std::string engineName{"argablarg"};
Manager::Manager(size_t stagingSize, SDL_Window* window) {
  vk::ApplicationInfo appInfo{appName.c_str(), 1, engineName.c_str(), 1,
                              VK_API_VERSION_1_1};
  // Validation layers are extremely helpful and don't incur that much
  // performance penalty, we'll only turn them off if we want absolute maximum
  // performance.
#ifdef NO_LAYERS
  const std::vector<const char*> layers;
#else
  const std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
  std::cout << "Running debug build\n";
#endif // DEBUG
  const std::vector<const char*> instanceExtensions = {"VK_KHR_surface",
                                                       "VK_KHR_xlib_surface"};
  const std::vector<const char*> deviceExtensions = {
      vk::KHRSwapchainExtensionName};
  vk::InstanceCreateInfo iCI(vk::InstanceCreateFlags(), &appInfo, layers,
                             instanceExtensions);
  try {
    instance = vk::createInstance(iCI);
  } catch (vk::SystemError& err) {
    std::cout << "Error: " << err.what() << std::endl;
    exit(-1);
  }
  physicalDevice = pickPhysicalDevice(instance);
  if (!SDL_Vulkan_CreateSurface(window, instance, nullptr,
                                bit_cast<VkSurfaceKHR*>(&surface))) {
    SDL_Log("Couldn't create surface: %s", SDL_GetError());
  }
  getQueueFamilyIndices(surface);
  float queuePriority = 1.0f;

  std::vector<vk::DeviceQueueCreateInfo> dQCI = {
      {vk::DeviceQueueCreateFlags(), cQFI, 1, &queuePriority}};
  if (gQFI != cQFI) {
    dQCI.emplace_back(vk::DeviceQueueCreateFlags(), gQFI, 1, &queuePriority);
  }
  if (pQFI != cQFI && pQFI != cQFI) {
    dQCI.emplace_back(vk::DeviceQueueCreateFlags(), pQFI, 1, &queuePriority);
  }
  vk::PhysicalDeviceFeatures phys_dev_features;
  phys_dev_features.shaderFloat64 = vk::True;
  phys_dev_features.shaderInt64 = vk::True;
  vk::DeviceCreateInfo dCI(vk::DeviceCreateFlags(), dQCI, {}, deviceExtensions,
                           &phys_dev_features, nullptr);
  device = physicalDevice.createDevice(dCI);
  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  cQFI);
  commandPool = device.createCommandPool(commandPoolCreateInfo);
  queue = device.getQueue(cQFI, 0);
  fence = device.createFence(vk::FenceCreateInfo());
  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.physicalDevice = physicalDevice;
  allocatorInfo.vulkanApiVersion = physicalDevice.getProperties().apiVersion;
  allocatorInfo.device = device;
  allocatorInfo.instance = instance;
  vmaCreateAllocator(&allocatorInfo, &allocator);
  vk::BufferCreateInfo stagingBCI({}, stagingSize,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst);
  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  stagingAllocation = VmaAllocation{};
  stagingInfo = VmaAllocationInfo{};
  vmaCreateBuffer(allocator, bit_cast<VkBufferCreateInfo*>(&stagingBCI),
                  &allocCreateInfo, bit_cast<VkBuffer*>(&staging),
                  &stagingAllocation, &stagingInfo);
  std::cout << "Finished creating Manager\n";
}

void Manager::copyBuffer(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                         uint32_t bufferSize) {
  auto commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  commandBuffer.copyBuffer(srcBuffer, dstBuffer,
                           vk::BufferCopy(0, 0, bufferSize));
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  queue.submit(submitInfo, fence);
  auto result = device.waitForFences(fence, true, -1);
  result = device.resetFences(1, &fence);
  device.freeCommandBuffers(commandPool, commandBuffer);
}

vk::CommandBuffer Manager::beginRecord(vk::CommandBufferUsageFlagBits bits) {
  auto commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI(bits);
  commandBuffer.begin(cBBI);

  return commandBuffer;
}

void Manager::writeToBuffer(MetaBuffer& dest, const void* source, size_t size) {
  // Catch if we're trying to write more data than the staging buffer can store.
  if (size > stagingInfo.size) {
    vmaDestroyBuffer(allocator, staging, stagingAllocation);
    vk::BufferCreateInfo stagingBCI({}, size,
                                    vk::BufferUsageFlagBits::eTransferSrc |
                                        vk::BufferUsageFlagBits::eTransferDst);
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    stagingAllocation = VmaAllocation{};
    stagingInfo = VmaAllocationInfo{};
    vmaCreateBuffer(allocator, bit_cast<VkBufferCreateInfo*>(&stagingBCI),
                    &allocCreateInfo, bit_cast<VkBuffer*>(&staging),
                    &stagingAllocation, &stagingInfo);
  }

  memcpy(stagingInfo.pMappedData, source, size);
  copyBuffer(staging, dest.buffer, size);
}

void Manager::writeFromBuffer(MetaBuffer& source, void* dest, size_t size) {
  // Catch if we're trying to write more data than the staging buffer can store.
  if (size > stagingInfo.size) {
    vmaDestroyBuffer(allocator, staging, stagingAllocation);
    vk::BufferCreateInfo stagingBCI({}, size,
                                    vk::BufferUsageFlagBits::eTransferSrc |
                                        vk::BufferUsageFlagBits::eTransferDst);
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    stagingAllocation = VmaAllocation{};
    stagingInfo = VmaAllocationInfo{};
    vmaCreateBuffer(allocator, bit_cast<VkBufferCreateInfo*>(&stagingBCI),
                    &allocCreateInfo, bit_cast<VkBuffer*>(&staging),
                    &stagingAllocation, &stagingInfo);
  }

  copyBuffer(source.buffer, staging, size);
  memcpy(dest, stagingInfo.pMappedData, size);
}

void Manager::execute(vk::CommandBuffer& b) {
  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &b);
  queue.submit(submitInfo, fence);
  auto result = device.waitForFences(fence, vk::True, -1);
  result = device.resetFences(1, &fence);
}

void Manager::executeNoSync(vk::CommandBuffer& b) {
  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &b);
  queue.submit(submitInfo);
}

void Manager::queueWaitIdle() { queue.waitIdle(); }

void Manager::getQueueFamilyIndices(vk::SurfaceKHR& surface) {
  auto queueFamilyProps = physicalDevice.getQueueFamilyProperties();
  for (u32 i = 0; i < queueFamilyProps.size(); i++) {
    if (queueFamilyProps[i].queueFlags & vk::QueueFlagBits::eGraphics) {
      gQFI = i;
    }
    if (queueFamilyProps[i].queueFlags & vk::QueueFlagBits::eCompute) {
      cQFI = i;
    }
    if (physicalDevice.getSurfaceSupportKHR(i, surface)) {
      pQFI = i;
    }
  }
  if (cQFI == UINT32_MAX) {
    runtime_exc("Fatal: Unable to find compute queue family index.");
  }
  if (gQFI == UINT32_MAX || pQFI == UINT32_MAX) {
    std::cout << "Warning: Unable to find graphics/present queue, not showing "
                 "any output."
              << std::endl;
  }
}

Algorithm Manager::makeAlgorithmRaw(std::string spirvname,
                                    std::vector<MetaBuffer*> buffers,
                                    const u8* specConsts, const u32* sizes,
                                    size_t nConsts, const u32* pushSizes,
                                    size_t nPushConstants) {
  const auto spirv = readFile<u32>(spirvname);
  return Algorithm(&device, buffers, spirv, specConsts, sizes, nConsts,
                   pushSizes, nPushConstants);
}

void appendOpNoBarrier(vk::CommandBuffer& b, Algorithm& a, u32 X, u32 Y,
                       u32 Z) {
  b.bindPipeline(vk::PipelineBindPoint::eCompute, a.m_Pipeline);
  b.bindDescriptorSets(vk::PipelineBindPoint::eCompute, a.m_PipelineLayout, 0,
                       a.m_DescriptorSet, nullptr);
  b.dispatch(X, Y, Z);
}

void appendOp(vk::CommandBuffer& b, Algorithm& a, u32 X, u32 Y, u32 Z) {
  b.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                    vk::PipelineStageFlagBits::eAllCommands, {},
                    fullMemoryBarrier, nullptr, nullptr);
  b.bindPipeline(vk::PipelineBindPoint::eCompute, a.m_Pipeline);
  b.bindDescriptorSets(vk::PipelineBindPoint::eCompute, a.m_PipelineLayout, 0,
                       a.m_DescriptorSet, nullptr);
  b.dispatch(X, Y, Z);
}

Manager::~Manager() {
  device.waitIdle();
  device.destroyFence(fence);
  vmaDestroyBuffer(allocator, staging, stagingAllocation);
  vmaDestroyAllocator(allocator);
  device.destroyCommandPool(commandPool);
  device.destroy();
  instance.destroySurfaceKHR(surface);
  instance.destroy();
}

Renderer::Renderer() {}
void Renderer::cleanupSwapchain() {
  vk::Device& dev = mgr->device;
  for (auto& fb : swapChainFrameBuffers) {
    dev.destroyFramebuffer(fb);
  }
  for (auto& iv : swapChainImageViews) {
    dev.destroyImageView(iv);
  }
  dev.destroySwapchainKHR(swapChain);
}
void Renderer::recreateSwapchain() {
  int width = 0, height = 0;
  while (width == 0 | height == 0) {
    SDL_GetWindowSize(window, &width, &height);
    SDL_Event event;
    SDL_WaitEvent(&event);
  };
  mgr->device.waitIdle();
  cleanupSwapchain();
};

vk::SurfaceFormatKHR
pickSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& formats) {
  assert(!formats.empty());
  vk::SurfaceFormatKHR pickedFormat = formats[0];
  if (formats.size() == 1) {
    if (formats[0].format == vk::Format::eUndefined) {
      pickedFormat.format = vk::Format::eB8G8R8A8Unorm;
      pickedFormat.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    }
  } else {
    // request several formats, the first found will be used
    vk::Format requestedFormats[] = {
        vk::Format::eB8G8R8A8Unorm, vk::Format::eR8G8B8A8Unorm,
        vk::Format::eB8G8R8Unorm, vk::Format::eR8G8B8Unorm};
    vk::ColorSpaceKHR requestedColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    for (size_t i = 0;
         i < sizeof(requestedFormats) / sizeof(requestedFormats[0]); i++) {
      vk::Format requestedFormat = requestedFormats[i];
      auto it = std::find_if(formats.begin(), formats.end(),
                             [requestedFormat, requestedColorSpace](
                                 vk::SurfaceFormatKHR const& f) {
                               return (f.format == requestedFormat) &&
                                      (f.colorSpace == requestedColorSpace);
                             });
      if (it != formats.end()) {
        pickedFormat = *it;
        break;
      }
    }
  }
  assert(pickedFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear);
  return pickedFormat;
}

void Renderer::createSwapChain(const vk::SwapchainKHR& oldSwapChain) {
  vk::SurfaceFormatKHR surfaceFormat =
      pickSurfaceFormat(mgr->physicalDevice.getSurfaceFormatsKHR(surface));
  swapChainImageFormat = surfaceFormat.format;
  vk::SurfaceCapabilitiesKHR surfaceCapabilities =
      mgr->physicalDevice.getSurfaceCapabilitiesKHR(surface);
  vk::Extent2D newExtent;
  if (surfaceCapabilities.currentExtent.width ==
      std::numeric_limits<uint32_t>::max()) {
    swapChainExtent.width = std::clamp(
        swapChainExtent.width, surfaceCapabilities.minImageExtent.width,
        surfaceCapabilities.maxImageExtent.width);
    swapChainExtent.height = std::clamp(
        swapChainExtent.height, surfaceCapabilities.minImageExtent.height,
        surfaceCapabilities.maxImageExtent.height);
  } else {
    newExtent = surfaceCapabilities.currentExtent;
  }
  vk::SurfaceTransformFlagBitsKHR preTransform =
      (surfaceCapabilities.supportedTransforms &
       vk::SurfaceTransformFlagBitsKHR::eIdentity)
          ? vk::SurfaceTransformFlagBitsKHR::eIdentity
          : surfaceCapabilities.currentTransform;
  vk::CompositeAlphaFlagBitsKHR compositeAlpha =
      (surfaceCapabilities.supportedCompositeAlpha &
       vk::CompositeAlphaFlagBitsKHR::ePreMultiplied)
          ? vk::CompositeAlphaFlagBitsKHR::ePreMultiplied
      : (surfaceCapabilities.supportedCompositeAlpha &
         vk::CompositeAlphaFlagBitsKHR::ePostMultiplied)
          ? vk::CompositeAlphaFlagBitsKHR::ePostMultiplied
      : (surfaceCapabilities.supportedCompositeAlpha &
         vk::CompositeAlphaFlagBitsKHR::eInherit)
          ? vk::CompositeAlphaFlagBitsKHR::eInherit
          : vk::CompositeAlphaFlagBitsKHR::eOpaque;
  vk::SwapchainCreateInfoKHR createInfo(
      {}, surface, surfaceCapabilities.minImageCount + 1, swapChainImageFormat,
      surfaceFormat.colorSpace, newExtent, 1,
      vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment |
                          vk::ImageUsageFlagBits::eTransferSrc),
      vk::SharingMode::eExclusive, {}, preTransform, compositeAlpha,
      vk::PresentModeKHR::eFifo, true, oldSwapChain);
  if (mgr->pQFI != mgr->gQFI) {
    uint32_t queueFamilyIndices[2] = {mgr->gQFI, mgr->pQFI};
    // If the graphics and present queues are from different queue families, we
    // either have to explicitly transfer ownership of images between the
    // queues, or we have to create the swapchain with imageSharingMode as
    // vk::SharingMode::eConcurrent
    createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  }
  swapChain = mgr->device.createSwapchainKHR(createInfo);
  swapChainImages = mgr->device.getSwapchainImagesKHR(swapChain);
  swapChainImageViews.resize(swapChainImages.size());
  vk::ImageViewCreateInfo iVCI({}, {}, vk::ImageViewType::e2D,
                               swapChainImageFormat, {},
                               {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  for (u32 i = 0; i < swapChainImageViews.size(); i++) {
    iVCI.image = swapChainImages[i];
    swapChainImageViews[i] = mgr->device.createImageView(iVCI);
  }
}

void Renderer::recordCommandBuffer(vk::CommandBuffer& cB, u32 imageIndex) {
  vk::CommandBufferBeginInfo cBBI;
  cB.begin(cBBI);
  std::array<vk::ClearValue, 2> clearValues;
  clearValues[0].setColor({0.2f, 0.2f, 0.2f, 1.0f});
  clearValues[1].setDepthStencil({1.0f, 0});
  vk::RenderPassBeginInfo rPBI(renderPass, swapChainFrameBuffers[imageIndex],
                               vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent),
                               clearValues);

  cB.beginRenderPass(rPBI, vk::SubpassContents::eInline);
  cB.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

  cB.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                        graphicsPipelineLayout, 0, descriptorSet, nullptr);
  cB.bindVertexBuffers(0, vertexBuffer.buffer, {0});
  cB.setViewport(0, vk::Viewport(0.0f, 0.0f, swapChainExtent.width,
                                 swapChainExtent.height, 0.0f, 1.0f));
  cB.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
  cB.draw(6, 1, 0, 0);
  cB.endRenderPass();
  cB.end();
}
void Renderer::drawFrame() {
  vk::Device& dev = mgr->device;
  if (dev.waitForFences(inFlightFences[currentFrame], vk::True, -1) !=
      vk::Result::eSuccess) {
    runtime_exc("Failed waiting for fences");
  };
  u32 imageIndex;
  vk::ResultValue<u32> res = dev.acquireNextImageKHR(
      swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);
  if (res.result == vk::Result::eErrorOutOfDateKHR) {
    recreateSwapchain();
    return;
  } else if (res.result != vk::Result::eSuccess &&
             res.result != vk::Result::eSuboptimalKHR) {
    runtime_exc("Failed to acquire swap chain image!");
  }

  dev.resetFences(inFlightFences[currentFrame]);
  vk::CommandBuffer& cB = commandBuffers[currentFrame];
  cB.reset();
  recordCommandBuffer(cB, imageIndex);

  vk::PipelineStageFlags waitDestinationStageMask(
      vk::PipelineStageFlagBits::eColorAttachmentOutput);
  vk::SubmitInfo submitInfo(imageAvailableSemaphores[currentFrame],
                            waitDestinationStageMask, cB,
                            renderFinishedSemaphores[currentFrame]);
  graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);
  auto result = presentQueue.presentKHR(
      vk::PresentInfoKHR(renderFinishedSemaphores[currentFrame], swapChain));
  if (result == vk::Result::eErrorOutOfDateKHR ||
      result == vk::Result::eSuboptimalKHR || frameBufferResized) {
    frameBufferResized = false;
    recreateSwapchain();
  } else if (result != vk::Result::eSuccess) {
    runtime_exc("Failed to present swap chain image!");
  }

  currentFrame = (currentFrame + 1) % maxFramesInFlight;
}

Renderer::~Renderer() {
  vk::Device& dev = mgr->device;
  dev.destroyPipeline(graphicsPipeline);
  dev.destroyPipelineLayout(graphicsPipelineLayout);
  dev.destroyRenderPass(renderPass, nullptr);
  for (size_t i = 0; i < maxFramesInFlight; i++) {
    dev.destroySemaphore(imageAvailableSemaphores[i]);
    dev.destroySemaphore(renderFinishedSemaphores[i]);
    dev.destroyFence(inFlightFences[i]);
  }
  mgr->instance.destroySurfaceKHR(surface);
}
