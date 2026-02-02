#include "mathhelpers.hpp"
#include <cmath>
#include <cstddef>
#include <format>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>
#define VMA_IMPLEMENTATION 1003000
// #include "colormaps.hpp"
#include "vk_mem_alloc.h"
#include "vkcore.hpp"
#include <cstdint>
#include <iostream>
#include <set>

constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
// static const char* BasePath = SDL_GetBasePath();

u32 get_compute_queue_family_index(vk::PhysicalDevice phys_dev) {
  auto queue_family_props = phys_dev.getQueueFamilyProperties();
  auto index =
      std::find_if(queue_family_props.begin(), queue_family_props.end(),
                   [](vk::QueueFamilyProperties qfp) {
                     return qfp.queueFlags & vk::QueueFlagBits::eCompute;
                   });
  return std::distance(queue_family_props.begin(), index);
}

std::array<u32, 2>
get_graphics_present_queue_family_indices(vk::PhysicalDevice phys_dev,
                                          vk::SurfaceKHR surface) {
  auto queueFamilyProps = phys_dev.getQueueFamilyProperties();
  u32 gQFI = UINT32_MAX;
  u32 pQFI = UINT32_MAX;
  for (u32 i = 0; i < queueFamilyProps.size(); i++) {
    if (queueFamilyProps[i].queueFlags & vk::QueueFlagBits::eGraphics) {
      gQFI = i;
    }
    if (phys_dev.getSurfaceSupportKHR(i, surface)) {
      pQFI = i;
    }
  }
  if (gQFI == UINT32_MAX) {
    runtime_exc("Fatal: Unable to find graphics queue family index.");
  }
  if (pQFI == UINT32_MAX) {
    runtime_exc("Fatal: Unable to find present queue family index.");
  }
  return {gQFI, pQFI};
}

void record_drawing_commands(
    vk::Framebuffer framebuffer, vk::RenderPass render_pass,
    vk::Extent2D swap_extent, vk::Pipeline graphics_pipeline,
    vk::PipelineLayout gpl, vk::DescriptorSet descriptor_set,
    vk::Buffer vertex_buffer, vk::CommandBuffer command_buffer) {
  /*vk::CommandBufferAllocateInfo cbAllocInfo(
      command_pool, vk::CommandBufferLevel::ePrimary, n_images);
  command_buffers.resize(n_images);
  vkAllocateCommandBuffers(dev,
                           pcast<VkCommandBufferAllocateInfo>(&cbAllocInfo),
                           pcast<VkCommandBuffer>(command_buffers.data()));*/
  vk::ClearValue clear{{1.0f, 1.0f, 1.0f, 1.0f}};
  vk::RenderPassBeginInfo render_pass_info(
      render_pass, framebuffer, vk::Rect2D(vk::Offset2D(0, 0), swap_extent),
      clear);
  vk::Viewport viewport(0.0f, 0.0f, (f32)swap_extent.width,
                        (f32)swap_extent.height, 0.0f, 1.0f);
  vk::Rect2D scissor(vk::Offset2D(0, 0), swap_extent);
  command_buffer.setViewport(0, viewport);
  command_buffer.setScissor(0, scissor);
  command_buffer.beginRenderPass(render_pass_info,
                                 vk::SubpassContents::eInline);
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                              graphics_pipeline);
  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, gpl, 0,
                                    descriptor_set, nullptr);
  command_buffer.bindVertexBuffers(0, vertex_buffer, {0});
  command_buffer.draw(6, 1, 0, 0);
  command_buffer.endRenderPass();
}

void record_nondestructive_parallel_reduction(vk::CommandBuffer cb,
                                              u32 org_size,
                                              Algorithm& reduction,
                                              Algorithm& first_reduction) {
  appendOp(cb, first_reduction, org_size / (64 * 2), 1, 1);
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                     vk::PipelineStageFlagBits::eAllCommands, {},
                     fullMemoryBarrier, nullptr, nullptr);
  cb.bindPipeline(vk::PipelineBindPoint::eCompute, reduction.m_Pipeline);
  cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                        reduction.m_PipelineLayout, 0,
                        reduction.m_DescriptorSet, nullptr);
  u32 n_iters = uintlog2(org_size);
  std::vector<u32> strides(n_iters);
  for (u32 i = 0; i < n_iters; i++) {
    strides[i] = pow(2, i);
    cb.pushConstants(reduction.m_PipelineLayout,
                     vk::ShaderStageFlagBits::eCompute, 0, 4, &strides[i]);
    cb.dispatch(
        std::clamp(((org_size + 1) / 2) / (64 * strides[i]), 1u, UINT32_MAX), 1,
        1);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                       vk::PipelineStageFlagBits::eAllCommands, {},
                       fullMemoryBarrier, nullptr, nullptr);
  }
}

vk::SurfaceFormatKHR
pickSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats) {
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

std::vector<vk::Framebuffer>
create_framebuffers(vk::Device device, std::vector<vk::ImageView> img_views,
                    vk::RenderPass render_pass, vk::Extent2D extent) {
  std::vector<vk::Framebuffer> frame_buffers(img_views.size());
  std::transform(img_views.begin(), img_views.end(), frame_buffers.begin(),
                 [&](vk::ImageView view) {
                   vk::FramebufferCreateInfo info(
                       {}, render_pass, view, extent.width, extent.height, 1);
                   return device.createFramebuffer(info);
                 });
  return frame_buffers;
}

vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& present_modes,
    vk::PresentModeKHR requested_mode = vk::PresentModeKHR::eFifo) {
  if (requested_mode == vk::PresentModeKHR::eFifo) {
    return vk::PresentModeKHR::eFifo;
  }
  for (const auto& availablePresentMode : present_modes) {
    if (availablePresentMode == requested_mode) {
      return availablePresentMode;
    }
  }

  std::cout << "Warning: Requested present mode is not available, defaulting "
               "to FIFO.\n";
  return vk::PresentModeKHR::eFifo;
}

MetaBuffer::MetaBuffer() {
  buffer = vk::Buffer{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
}

MetaBuffer::MetaBuffer(VmaAllocator& allocator,
                       VmaAllocationCreateInfo& allocCreateInfo,
                       vk::BufferCreateInfo& BCI) {
  p_allocator = &allocator;
  buffer = vk::Buffer{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
  vmaCreateBuffer(allocator, pcast<VkBufferCreateInfo>(&BCI), &allocCreateInfo,
                  pcast<VkBuffer>(&buffer), &allocation, &aInfo);
}

void MetaBuffer::allocate(VmaAllocator& allocator,
                          VmaAllocationCreateInfo& allocCreateInfo,
                          vk::BufferCreateInfo& BCI) {
  p_allocator = &allocator;
  vmaCreateBuffer(allocator, pcast<VkBufferCreateInfo>(&BCI), &allocCreateInfo,
                  pcast<VkBuffer>(&buffer), &allocation, &aInfo);
}

MetaBuffer::~MetaBuffer() {
  vmaDestroyBuffer(*p_allocator, buffer, allocation);
}

AllocatedImage::AllocatedImage() {
  img = vk::Image{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
}

AllocatedImage::AllocatedImage(VmaAllocator& allocator,
                               VmaAllocationCreateInfo& allocCreateInfo,
                               vk::ImageCreateInfo& BCI) {
  p_allocator = &allocator;
  img = vk::Image{};
  allocation = VmaAllocation{};
  aInfo = VmaAllocationInfo{};
  vmaCreateImage(allocator, pcast<VkImageCreateInfo>(&BCI), &allocCreateInfo,
                 pcast<VkImage>(&img), &allocation, &aInfo);
}

void AllocatedImage::allocate(VmaAllocator& allocator,
                              VmaAllocationCreateInfo& allocCreateInfo,
                              vk::ImageCreateInfo& BCI) {
  p_allocator = &allocator;
  vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&BCI),
                 &allocCreateInfo, reinterpret_cast<VkImage*>(&img),
                 &allocation, &aInfo);
}

AllocatedImage::~AllocatedImage() {
  vmaDestroyImage(*p_allocator, img, allocation);
}

Algorithm::Algorithm(vk::Device device, u32 img_views, u32 buffers, u32 n_ubo,
                     const std::vector<u32>& spirv, const u8* specConsts,
                     const size_t* sizes, size_t nConsts,
                     const size_t* pushSizes, size_t nPushConstants) {
  initialize(device, img_views, buffers, n_ubo, spirv, specConsts, sizes,
             nConsts, pushSizes, nPushConstants);
}

Algorithm::~Algorithm() {
  m_device.destroyDescriptorSetLayout(m_DSL);
  m_device.destroyDescriptorPool(m_DescriptorPool);
  m_device.destroyShaderModule(m_ShaderModule);
  m_device.destroyPipeline(m_Pipeline);
  m_device.destroyPipelineLayout(m_PipelineLayout);
}

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
    runtime_exc("Couldn't enumerate instance extension properties.\n");
  }

  std::vector<vk::ExtensionProperties> extensionProperties(count);

  // Get the extensions
  result = vk::enumerateInstanceExtensionProperties(nullptr, &count,
                                                    extensionProperties.data());
  if (result != vk::Result::eSuccess) {
    runtime_exc("Couldn't write instance extension properties to buffer.\n");
  }

  std::set<std::string> extensions;
  for (auto& extension : extensionProperties) {
    extensions.insert(extension.extensionName);
  }

  return extensions;
}

static const std::string appName{"Vulkan GPE Simulator"};
static const std::string engineName{"argablarg"};
Manager::Manager(size_t stagingSize /*,  SDL_Window* _window */) {
  // if (_window) {
  //   window = _window;
  // }
  vk::ApplicationInfo appInfo{appName.c_str(), 1, engineName.c_str(), 1,
                              VK_API_VERSION_1_3};
  // Validation layers are extremely helpful, we'll only turn them off if we
  // want absolute maximum performance.
#ifdef NO_LAYERS
  const std::vector<const char*> layers;
#else
  const std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
  std::cout << "Running debug build\n";
#endif // DEBUG
  u32 instance_extension_count = 0;
  char const* const* instance_extensions = nullptr;
  /*char const* const* instance_extensions = [&]() {
    if (window) {
      return SDL_Vulkan_GetInstanceExtensions(&instance_extension_count);
    }
    return (char const* const*)nullptr;
  }();*/
  // const std::vector<const char*> instanceExtensions = ;
  const std::vector<const char*> deviceExtensions{};
  /* const std::vector<const char*> deviceExtensions = [&]() {
    if (window) {
      return std::vector<const char*>{vk::KHRSwapchainExtensionName};
    }
    return std::vector<const char*>{};
  }(); */
  vk::InstanceCreateInfo iCI(vk::InstanceCreateFlags(), &appInfo, layers.size(),
                             layers.data(), instance_extension_count,
                             instance_extensions);
  try {
    instance = vk::createInstance(iCI);
  } catch (vk::SystemError& err) {
    std::cout << "Error: " << err.what() << std::endl;
    exit(-1);
  }
  physicalDevice = pickPhysicalDevice(instance);

  std::vector<u32> qfis;
  /*if (window) {
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr,
                                  pcast<VkSurfaceKHR>(&surface))) {
      SDL_Log("CreateSurface failed with error: %s", SDL_GetError());
    }
  }*/
  qfis.push_back(get_compute_queue_family_index(physicalDevice));
  /*if (window) {
    auto gpqfis =
        get_graphics_present_queue_family_indices(physicalDevice, surface);
    if (gpqfis[0] == gpqfis[1]) {
      if (qfis[0] != gpqfis[0]) {
        qfis.push_back(gpqfis[0]);
      }
    } else {
      if (qfis[0] != gpqfis[0]) {
        qfis.push_back(gpqfis[0]);
      }
      if (qfis[0] != gpqfis[1]) {
        qfis.push_back(gpqfis[1]);
      }
    }
  }*/
  cQFI = qfis[0];

  float queuePriority = 1.0f;

  std::vector<vk::DeviceQueueCreateInfo> dQCI(qfis.size());
  std::transform(qfis.begin(), qfis.end(), dQCI.begin(), [&](u32 qfi) {
    return vk::DeviceQueueCreateInfo({}, qfi, 1, &queuePriority);
  });
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
  vk::BufferCreateInfo stagingBCI({}, round_up_x16(stagingSize),
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

void copyBufferNow(vk::Device device, vk::CommandPool commandPool,
                   vk::Queue queue, vk::Fence fence, vk::Buffer& srcBuffer,
                   vk::Buffer& dstBuffer, u32 bufferSize, u32 src_offset,
                   u32 dst_offset) {
  auto commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  commandBuffer.copyBuffer(srcBuffer, dstBuffer,
                           vk::BufferCopy(src_offset, dst_offset, bufferSize));
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  queue.submit(submitInfo, fence);
  auto result = device.waitForFences(fence, true, -1);
  result = device.resetFences(1, &fence);
  device.freeCommandBuffers(commandPool, commandBuffer);
}

vk::CommandBuffer Manager::copyOp(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                                  u32 bufferSize, u32 src_offset,
                                  u32 dst_offset) {
  auto commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI{};
  commandBuffer.begin(cBBI);
  commandBuffer.copyBuffer(srcBuffer, dstBuffer,
                           vk::BufferCopy(src_offset, dst_offset, bufferSize));
  commandBuffer.end();
  return commandBuffer;
}

void Manager::copyBuffer(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                         u32 bufferSize, u32 src_offset, u32 dst_offset) {
  copyBufferNow(device, commandPool, queue, fence, srcBuffer, dstBuffer,
                bufferSize, src_offset, dst_offset);
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

void Manager::writeToBuffer(MetaBuffer& dest, const void* source, size_t size,
                            size_t src_offset, size_t dst_offset) {
  // Catch if we're trying to write more data than the staging buffer can
  // store.
  if (size > stagingInfo.size) {
    vmaDestroyBuffer(allocator, staging, stagingAllocation);
    vk::BufferCreateInfo stagingBCI({}, round_up_x16(size),
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
  copyBuffer(staging, dest.buffer, size, src_offset, dst_offset);
}

void Manager::writeFromBuffer(MetaBuffer& source, void* dest, size_t size) {
  // Catch if we're trying to write more data than the staging buffer can
  // store.
  if (size > stagingInfo.size) {
    vmaDestroyBuffer(allocator, staging, stagingAllocation);
    vk::BufferCreateInfo stagingBCI({}, round_up_x16(size),
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

Algorithm Manager::makeAlgorithmRaw(std::string spirvName, u32 nImgViews,
                                    u32 nBuffers, u32 nUBOs,
                                    const u8* specConsts, const size_t* sizes,
                                    size_t nConsts, const size_t* pushSizes,
                                    size_t nPushConstants) {
  const auto spirv = readFile<u32>(spirvName);
  return Algorithm(device, nImgViews, nBuffers, nUBOs, spirv, specConsts, sizes,
                   nConsts, pushSizes, nPushConstants);
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
  /*if (window != nullptr) {
    instance.destroySurfaceKHR(surface);
  }*/
  vmaDestroyBuffer(allocator, staging, stagingAllocation);
  vmaDestroyAllocator(allocator);
  device.destroyCommandPool(commandPool);
  device.destroy();
  instance.destroy();
}

vk::SwapchainKHR create_swapchain(vk::Device device, vk::SurfaceKHR surface,
                                  vk::SurfaceCapabilitiesKHR capabilities,
                                  vk::SurfaceFormatKHR surface_format,
                                  vk::PresentModeKHR present_mode,
                                  vk::Extent2D swapchain_extent, u32 n_images,
                                  std::array<u32, 2> qfis,
                                  vk::SwapchainKHR old_swapchain = nullptr) {
  vk::SurfaceTransformFlagBitsKHR pre_transform =
      (capabilities.supportedTransforms &
       vk::SurfaceTransformFlagBitsKHR::eIdentity)
          ? vk::SurfaceTransformFlagBitsKHR::eIdentity
          : capabilities.currentTransform;

  vk::CompositeAlphaFlagBitsKHR composite_alpha =
      (capabilities.supportedCompositeAlpha &
       vk::CompositeAlphaFlagBitsKHR::ePreMultiplied)
          ? vk::CompositeAlphaFlagBitsKHR::ePreMultiplied
      : (capabilities.supportedCompositeAlpha &
         vk::CompositeAlphaFlagBitsKHR::ePostMultiplied)
          ? vk::CompositeAlphaFlagBitsKHR::ePostMultiplied
      : (capabilities.supportedCompositeAlpha &
         vk::CompositeAlphaFlagBitsKHR::eInherit)
          ? vk::CompositeAlphaFlagBitsKHR::eInherit
          : vk::CompositeAlphaFlagBitsKHR::eOpaque;
  vk::SwapchainCreateInfoKHR create_info(
      {}, surface, n_images, surface_format.format, surface_format.colorSpace,
      swapchain_extent, 1, vk::ImageUsageFlagBits::eColorAttachment,
      vk::SharingMode::eExclusive, {}, pre_transform, composite_alpha,
      present_mode, true, old_swapchain);
  if (qfis[0] != qfis[1]) {
    // If the graphics and present queues are from different queue families,
    // we either have to explicitly transfer ownership of images between the
    // queues, or we have to create the swapchain with imageSharingMode as
    // VK_SHARING_MODE_CONCURRENT
    create_info.imageSharingMode = vk::SharingMode::eConcurrent;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = qfis.data();
  }
  return device.createSwapchainKHR(create_info);
}

std::vector<vk::ImageView>
create_image_views(vk::Device device, vk::SurfaceFormatKHR surface_format,
                   std::vector<vk::Image> images) {
  std::vector<vk::ImageView> swapchain_image_views;
  swapchain_image_views.reserve(images.size());
  vk::ImageViewCreateInfo img_view_create_info(
      {}, {}, vk::ImageViewType::e2D, surface_format.format, {},
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  for (auto img : images) {
    img_view_create_info.image = img;
    swapchain_image_views.push_back(
        device.createImageView(img_view_create_info));
  }
  return swapchain_image_views;
}

void Algorithm::initialize(vk::Device device, u32 n_imgs, u32 n_buffers,
                           u32 n_ubo, const std::vector<u32>& spirv,
                           const u8* specConsts, const size_t* sizes,
                           size_t nConsts, const size_t* pushSizes,
                           size_t nPushConstants) {
  m_device = device;
  vk::ShaderModuleCreateInfo shaderMCI(vk::ShaderModuleCreateFlags(), spirv);
  m_ShaderModule = device.createShaderModule(shaderMCI);
  std::vector<vk::DescriptorSetLayoutBinding> dSLBs(n_imgs + n_buffers + n_ubo);
  {
    u32 i = 0;
    for (; i < n_imgs; i++) {
      dSLBs[i] = {i, vk::DescriptorType::eStorageImage, 1,
                  vk::ShaderStageFlagBits::eCompute};
    }
    for (; i < n_buffers + n_imgs; i++) {
      dSLBs[i] = {i, vk::DescriptorType::eStorageBuffer, 1,
                  vk::ShaderStageFlagBits::eCompute};
    }
    for (; i < n_buffers + n_imgs + n_ubo; i++) {
      dSLBs[i] = {i, vk::DescriptorType::eUniformBuffer, 1,
                  vk::ShaderStageFlagBits::eCompute};
    }
  }
  vk::DescriptorSetLayoutCreateInfo dSLCI(vk::DescriptorSetLayoutCreateFlags(),
                                          dSLBs);
  m_DSL = device.createDescriptorSetLayout(dSLCI);
  std::vector<vk::PushConstantRange> ranges(nPushConstants);
  u32 pushOffsets = 0;
  for (u32 i = 0; i < nPushConstants; i++) {
    ranges[i] = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute,
                                      pushOffsets, pushSizes[i]);
    pushOffsets += pushSizes[i];
  }
  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(), m_DSL,
                                    ranges);
  m_PipelineLayout = device.createPipelineLayout(pLCI);
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
  auto result = device.createComputePipeline({}, cPCI);
  m_Pipeline = result.value;

  // This is probably not the most efficient way to do this, but I'm not going
  // to mess around with the descriptors after creation so the only overhead
  // should be memory, and I'm not going to make thousands of these so
  // it should be fine.
  std::vector<vk::DescriptorPoolSize> dPSes;
  if (n_imgs > 0) {
    dPSes.emplace_back(vk::DescriptorType::eStorageImage, n_imgs);
  }
  if (n_buffers > 0) {
    dPSes.emplace_back(vk::DescriptorType::eStorageBuffer, n_buffers);
  }
  if (n_ubo > 0) {
    dPSes.emplace_back(vk::DescriptorType::eUniformBuffer, n_ubo);
  }
  vk::DescriptorPoolCreateInfo dPCI(
      vk::DescriptorPoolCreateFlags(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
      1, dPSes);
  m_DescriptorPool = device.createDescriptorPool(dPCI);
  vk::DescriptorSetAllocateInfo dSAI(m_DescriptorPool, 1, &m_DSL);
  auto descriptorSets = device.allocateDescriptorSets(dSAI);
  m_DescriptorSet = descriptorSets[0];
}

void Algorithm::bindData(const std::vector<vk::ImageView>& img_views,
                         const std::vector<MetaBuffer*>& buffers,
                         const std::vector<MetaBuffer*>& ubos) {
  std::vector<vk::DescriptorImageInfo> dIIs(img_views.size());
  std::vector<vk::DescriptorBufferInfo> dBIs(buffers.size());
  std::vector<vk::DescriptorBufferInfo> dUBIs(ubos.size());
  for (size_t i = 0; i < dIIs.size(); i++) {
    dIIs[i] = {{}, img_views[i], vk::ImageLayout::eGeneral};
  }
  for (size_t i = 0; i < dBIs.size(); i++) {
    dBIs[i] =
        vk::DescriptorBufferInfo(buffers[i]->buffer, 0, buffers[i]->aInfo.size);
  }
  for (size_t i = 0; i < dUBIs.size(); i++) {
    dBIs[i] = vk::DescriptorBufferInfo(ubos[i]->buffer, 0, ubos[i]->aInfo.size);
  }

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets(
      dIIs.size() + dBIs.size() + dUBIs.size());
  {
    u32 i = 0;
    for (; i < dIIs.size(); i++) {
      writeDescriptorSets[i] = {m_DescriptorSet, i, 0,
                                vk::DescriptorType::eStorageImage, dIIs[i]};
    }
    for (; i < dIIs.size() + dBIs.size(); i++) {
      writeDescriptorSets[i] = {m_DescriptorSet,
                                i,
                                0,
                                1,
                                vk::DescriptorType::eStorageBuffer,
                                nullptr,
                                &dBIs[i - dIIs.size()]};
    }
    for (; i < dIIs.size() + dBIs.size() + dUBIs.size(); i++) {
      writeDescriptorSets[i] = {m_DescriptorSet,
                                i,
                                0,
                                1,
                                vk::DescriptorType::eUniformBuffer,
                                nullptr,
                                &dUBIs[i - dIIs.size() - dBIs.size()]};
    }
  }
  m_device.updateDescriptorSets(writeDescriptorSets, {});
}

/* Renderer::Renderer(Manager& manager, u32 nx, u32 ny) {
  p_mgr = &manager;
  render_queue_indices = get_graphics_present_queue_family_indices(
      manager.physicalDevice, manager.surface);
  if (render_queue_indices[0] != render_queue_indices[1]) {
    runtime_exc("Graphics queue is different from present queue, I don't want "
                "to deal with that right now!");
  }
  graphicsQueue = manager.device.getQueue(render_queue_indices[0], 0);
  presentQueue = manager.device.getQueue(render_queue_indices[1], 0);
  capabilities =
      manager.physicalDevice.getSurfaceCapabilitiesKHR(manager.surface);
  auto formats = manager.physicalDevice.getSurfaceFormatsKHR(manager.surface);
  if (formats.empty()) {
    runtime_exc("Fatal: No surface formats found");
  }
  surface_format = pickSurfaceFormat(formats);
  auto presentModes =
      manager.physicalDevice.getSurfacePresentModesKHR(manager.surface);
  if (presentModes.empty()) {
    runtime_exc("Fatal: No present modes found");
  }
  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  render_queue_indices[0]);
  command_pool = manager.device.createCommandPool(commandPoolCreateInfo);

  present_mode = chooseSwapPresentMode(presentModes);
  if (capabilities.currentExtent.width ==
      std::numeric_limits<uint32_t>::max()) {
    s32 width, height;
    SDL_GetWindowSize(manager.window, &width, &height);
    u32 uwidth = (u32)width;
    u32 uheight = (u32)height;
    swapChainExtent.width =
        std::clamp(uwidth, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.height);
    swapChainExtent.height =
        std::clamp(uheight, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

  } else {
    swapChainExtent.width = std::clamp(capabilities.currentExtent.width,
                                       capabilities.minImageExtent.width,
                                       capabilities.maxImageExtent.width);
    swapChainExtent.height = std::clamp(capabilities.currentExtent.height,
                                        capabilities.minImageExtent.height,
                                        capabilities.maxImageExtent.height);
  }

  n_images = capabilities.maxImageCount == 0
                 ? 3u
                 : std::clamp(3u, capabilities.minImageCount,
                              capabilities.minImageCount);
  vk::SurfaceTransformFlagBitsKHR pre_transform =
      (capabilities.supportedTransforms &
       vk::SurfaceTransformFlagBitsKHR::eIdentity)
          ? vk::SurfaceTransformFlagBitsKHR::eIdentity
          : capabilities.currentTransform;
  capabilities.currentTransform = pre_transform;

  swapchain = create_swapchain(manager.device, manager.surface, capabilities,
                               surface_format, present_mode, swapChainExtent,
                               n_images, render_queue_indices);

  swapChainImages = manager.device.getSwapchainImagesKHR(swapchain);
  swapChainImageViews.reserve(swapChainImages.size());
  vk::ImageViewCreateInfo img_view_create_info(
      {}, {}, vk::ImageViewType::e2D, surface_format.format, {},
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  for (auto img : swapChainImages) {
    img_view_create_info.image = img;
    swapChainImageViews.push_back(
        manager.device.createImageView(img_view_create_info));
  }

  vk::AttachmentDescription color_attachment(
      vk::AttachmentDescriptionFlags(), surface_format.format,
      vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
      vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
      vk::ImageLayout::ePresentSrcKHR);

  vk::AttachmentReference color_attachment_ref(
      0, vk::ImageLayout::eColorAttachmentOptimal);

  vk::SubpassDescription subpass(vk::SubpassDescriptionFlags(),
                                 vk::PipelineBindPoint::eGraphics, {},
                                 color_attachment_ref);

  vk::SubpassDependency dependency{};
  dependency.setSrcSubpass(vk::SubpassExternal);
  dependency.setDstSubpass(0);
  dependency.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  dependency.setSrcAccessMask(vk::AccessFlagBits::eNone);
  dependency.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  dependency.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                              vk::AccessFlagBits::eColorAttachmentWrite);

  vk::RenderPassCreateInfo render_pass_info({}, color_attachment, subpass,
                                            dependency);

  renderPass = manager.device.createRenderPass(render_pass_info);
  swapChainFrameBuffers = create_framebuffers(
      manager.device, swapChainImageViews, renderPass, swapChainExtent);

  auto vert_code =
      readFile<u32>(std::string(BasePath) + "Shaders/quad.vert.spv");
  auto frag_code =
      readFile<u32>(std::string(BasePath) + "Shaders/quad.frag.spv");

  vk::ShaderModuleCreateInfo vert_MCI(vk::ShaderModuleCreateFlags(), vert_code);
  vk::ShaderModule vert_module = manager.device.createShaderModule(vert_MCI);
  vk::ShaderModuleCreateInfo frag_MCI(vk::ShaderModuleCreateFlags(), frag_code);
  vk::ShaderModule frag_module = manager.device.createShaderModule(frag_MCI);
  if (vert_module == VK_NULL_HANDLE) {
    runtime_exc("Failed to create vertex module");
  }
  if (frag_module == VK_NULL_HANDLE) {
    runtime_exc("Failed to create frag module");
  }

  vk::PipelineShaderStageCreateInfo vert_stage_info(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
      vert_module, "main");

  vk::PipelineShaderStageCreateInfo frag_stage_info(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment,
      frag_module, "main");
  vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_stage_info,
                                                       frag_stage_info};

  auto vertex_binding = PositionTextureVertex::bindingDscr();
  auto attr = PositionTextureVertex::attributeDscr();
  vk::PipelineVertexInputStateCreateInfo vertex_input_info(
      vk::PipelineVertexInputStateCreateFlags(), vertex_binding, attr);

  vk::PipelineInputAssemblyStateCreateInfo input_assembly(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::eTriangleList);
  vk::Viewport viewport(0.0, 0.0, (f32)swapChainExtent.width,
                        (f32)swapChainExtent.height, 0.0f, 1.0f);

  vk::Rect2D scissor(vk::Offset2D(0, 0), swapChainExtent);

  vk::PipelineViewportStateCreateInfo viewport_state(
      vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissor);
  vk::PipelineRasterizationStateCreateInfo rasterizer(
      vk::PipelineRasterizationStateCreateFlags(),
      false,                       // depthClampEnable
      false,                       // rasterizerDiscardEnable
      vk::PolygonMode::eFill,      // polygonMode
      vk::CullModeFlagBits::eBack, // cullMode
      vk::FrontFace::eClockwise,   // frontFace
      false,                       // depthBiasEnable
      0.0f,                        // depthBiasConstantFactor
      0.0f,                        // depthBiasClamp
      0.0f,                        // depthBiasSlopeFactor
      1.0f                         // lineWidth
  );

  vk::PipelineMultisampleStateCreateInfo multisampling(
      vk::PipelineMultisampleStateCreateFlags(), // flags
      vk::SampleCountFlagBits::e1                // rasterizationSamples
                                                 // other values can be default
  );

  vk::ColorComponentFlags colorComponentFlags(
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendAttachmentState colorBlendAttachment(
      false,                  // blendEnable
      vk::BlendFactor::eZero, // srcColorBlendFactor
      vk::BlendFactor::eZero, // dstColorBlendFactor
      vk::BlendOp::eAdd,      // colorBlendOp
      vk::BlendFactor::eZero, // srcAlphaBlendFactor
      vk::BlendFactor::eZero, // dstAlphaBlendFactor
      vk::BlendOp::eAdd,      // alphaBlendOp
      colorComponentFlags     // colorWriteMask
  );

  vk::PipelineColorBlendStateCreateInfo color_blending(
      vk::PipelineColorBlendStateCreateFlags(), // flags
      false,                                    // logicOpEnable
      vk::LogicOp::eCopy,                       // logicOp
      colorBlendAttachment,                     // attachments
      {{1.0f, 1.0f, 1.0f, 1.0f}}                // blendConstants
  );

  vk::DescriptorSetLayoutBinding dsl_binding(
      0, vk::DescriptorType::eCombinedImageSampler, 1,
      vk::ShaderStageFlagBits::eFragment);
  descriptorSetLayout = manager.device.createDescriptorSetLayout(
      vk::DescriptorSetLayoutCreateInfo({}, dsl_binding));
  vk::PipelineLayoutCreateInfo graphics_pipeline_layout_info{};
  graphics_pipeline_layout_info.setSetLayouts(descriptorSetLayout);

  graphicsPipelineLayout =
      manager.device.createPipelineLayout(graphics_pipeline_layout_info);

  std::vector<vk::DynamicState> dynamic_states = {vk::DynamicState::eViewport,
                                                  vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamic_info(
      vk::PipelineDynamicStateCreateFlags(), dynamic_states);

  vk::GraphicsPipelineCreateInfo pipeline_info(
      vk::PipelineCreateFlags(), shader_stages, &vertex_input_info,
      &input_assembly, nullptr, &viewport_state, &rasterizer, &multisampling,
      nullptr, &color_blending, &dynamic_info, graphicsPipelineLayout,
      renderPass);

  auto result = manager.device.createGraphicsPipeline({}, pipeline_info);
  assert(result.result == vk::Result::eSuccess);
  graphicsPipeline = result.value;
  manager.device.destroyShaderModule(frag_module, nullptr);
  manager.device.destroyShaderModule(vert_module, nullptr);

  std::vector<PositionTextureVertex> vertices = {
      {{-1.0f, -1.0f}, {0.0f, 0.0f}}, {{0.8f, -1.0f}, {1.0f, 0.0f}},
      {{0.8f, 1.0f}, {1.0f, 1.0f}},   {{0.8f, 1.0f}, {1.0f, 1.0f}},
      {{-1.0f, 1.0f}, {0.0f, 1.0f}},  {{-1.0f, -1.0f}, {0.0f, 0.0f}},
      {{0.8f, -1.0f}, {0.0f, 0.0f}},  {{1.0f, -1.0f}, {0.0f, 0.0f}},
      {{1.0f, 1.0f}, {0.0f, 1.0f}},   {{0.8f, -1.0f}, {0.0f, 0.0f}},
      {{0.8f, 1.0f}, {0.0f, 1.0f}},   {{1.0f, -1.0f}, {0.0f, 1.0f}}};
  vk::BufferCreateInfo vertexBCI(
      {}, vertices.size() * sizeof(PositionTextureVertex),
      vk::BufferUsageFlagBits::eVertexBuffer |
          vk::BufferUsageFlagBits::eTransferDst);
  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocCreateInfo.priority = 1.0f;
  vertexBuffer.allocate(manager.allocator, allocCreateInfo, vertexBCI);
  manager.writeToBuffer(vertexBuffer, vertices);

  vk::Format colormap_format = vk::Format::eR32G32B32A32Sfloat;
  vk::ImageCreateInfo colormap_img_info(
      {}, vk::ImageType::e2D, colormap_format, vk::Extent3D(nx, ny, 1), 1, 1,
      vk::SampleCountFlagBits::e1, vk::ImageTiling::eLinear,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
  VmaAllocationCreateInfo img_alloc_create_info{};
  img_alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
  img_alloc_create_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocCreateInfo.priority = 1.0f;
  colormap_img.allocate(manager.allocator, img_alloc_create_info,
                        colormap_img_info);
  vk::ImageMemoryBarrier initialbarrier{};
  initialbarrier.setOldLayout(vk::ImageLayout::eUndefined);
  // On Nvidia there is really only the General image layout. On AMD General
  // is compressed so you need to switch to a different layout for adequate
  // performance there
  initialbarrier.setNewLayout(vk::ImageLayout::eGeneral);
  initialbarrier.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
  initialbarrier.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
  initialbarrier.setImage(colormap_img.img);
  initialbarrier.setSubresourceRange(
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  initialbarrier.setSrcAccessMask({});
  initialbarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  oneTimeSubmit(manager.device, manager.commandPool, manager.queue,
                [&](vk::CommandBuffer b) {
                  b.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    {}, nullptr, nullptr, initialbarrier);
                });

  vk::ImageViewCreateInfo view_info(
      {}, colormap_img.img, vk::ImageViewType::e2D, colormap_format, {},
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  colormap_view = manager.device.createImageView(view_info);

  vk::BufferCreateInfo colormapBCI({}, 256 * sizeof(cm::AlignedColor),
                                   vk::BufferUsageFlagBits::eStorageBuffer |
                                       vk::BufferUsageFlagBits::eTransferDst |
                                       vk::BufferUsageFlagBits::eTransferSrc);
  colormap.allocate(manager.allocator, img_alloc_create_info, colormapBCI);
  manager.writeToBuffer(colormap, cm::viridis.data(),
                        cm::viridis.size() * sizeof(cm::AlignedColor));
  vk::BufferCreateInfo valueBCI({}, ny * nx * sizeof(f32),
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst |
                                    vk::BufferUsageFlagBits::eTransferSrc);
  const u32 n_target = (ny * nx) / 16;
  vk::BufferCreateInfo minmaxBCI({}, n_target * sizeof(f32),
                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                     vk::BufferUsageFlagBits::eTransferDst |
                                     vk::BufferUsageFlagBits::eTransferSrc);
  value_buffer.allocate(manager.allocator, img_alloc_create_info, valueBCI);
  minmax_buffer.allocate(manager.allocator, img_alloc_create_info, minmaxBCI);
  std::vector<f32> values(ny * nx);
  for (u32 j = 0; j < ny; j++) {
    for (u32 i = 0; i < nx; i++) {
      values[nx * j + i] = (f32)(i + j);
    }
  }
  manager.writeToBuffer(value_buffer, values);

  auto first_minmax_code =
      readFile<u32>(std::string(BasePath) + "Shaders/firstminmax.spv");
  auto minmax_code =
      readFile<u32>(std::string(BasePath) + "Shaders/minmax.spv");
  const size_t reduction_spec_sizes = 4;
  first_minmax_reduction.initialize(
      manager.device, {}, {&value_buffer, &minmax_buffer}, first_minmax_code,
      pcast<u8>(&n_target), &reduction_spec_sizes, 1);
  minmax_reduction.initialize(manager.device, {}, {&minmax_buffer}, minmax_code,
                              pcast<u8>(&n_target), &reduction_spec_sizes, 1);

  auto fill_code =
      readFile<u32>(std::string(BasePath) + "Shaders/colormap.spv");
  fill_colormap_img.initialize(p_mgr->device, {colormap_view},
                               {&colormap, &minmax_buffer, &value_buffer},
                               fill_code, pcast<u8>(&n_target),
                               &reduction_spec_sizes, 1);

  vk::SamplerCreateInfo colormap_sampler_info(
      vk::SamplerCreateFlags(), vk::Filter::eNearest, vk::Filter::eNearest,
      vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.0f, false, 1.0f, false,
      vk::CompareOp::eNever, 0.0f, 0.0f, vk::BorderColor::eFloatOpaqueWhite);
  colormap_sampler = manager.device.createSampler(colormap_sampler_info);

  vk::DescriptorPoolSize pool_size(vk::DescriptorType::eCombinedImageSampler,
                                   MAX_FRAMES_IN_FLIGHT);
  descriptorPool = manager.device.createDescriptorPool(
      {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, n_images, 1,
       &pool_size});

  descriptorSet =
      manager.device
          .allocateDescriptorSets({descriptorPool, descriptorSetLayout})
          .front();
  vk::DescriptorImageInfo descriptor_img_info(
      colormap_sampler, colormap_view, vk::ImageLayout::eShaderReadOnlyOptimal);
  vk::WriteDescriptorSet descriptor_write(
      descriptorSet, 0, 0, vk::DescriptorType::eCombinedImageSampler,
      descriptor_img_info);
  manager.device.updateDescriptorSets(descriptor_write, {});

  imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
  renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
  inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
  imageInFlightFences.resize(swapChainImages.size(), VK_NULL_HANDLE);

  vk::SemaphoreCreateInfo semaphore_info{};

  vk::FenceCreateInfo fence_info(vk::FenceCreateFlagBits::eSignaled);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    imageAvailableSemaphores[i] =
        manager.device.createSemaphore(semaphore_info);
    renderFinishedSemaphores[i] =
        manager.device.createSemaphore(semaphore_info);
    inFlightFences[i] = manager.device.createFence(fence_info);
  }

  vk::CommandBufferAllocateInfo cbAllocInfo(
      command_pool, vk::CommandBufferLevel::ePrimary, n_images);
  commandBuffers.resize(n_images);
  vkAllocateCommandBuffers(p_mgr->device,
                           pcast<VkCommandBufferAllocateInfo>(&cbAllocInfo),
                           pcast<VkCommandBuffer>(commandBuffers.data()));
  vk::CommandBufferBeginInfo begin_info{};
  reduction_buffer = manager.beginRecord();
  vk::ImageMemoryBarrier present_to_storage{};
  present_to_storage.setNewLayout(vk::ImageLayout::eGeneral);
  present_to_storage.setOldLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
  present_to_storage.setSrcQueueFamilyIndex(manager.cQFI);
  present_to_storage.setDstQueueFamilyIndex(manager.cQFI);
  present_to_storage.setDstAccessMask(vk::AccessFlagBits::eShaderWrite);
  present_to_storage.setSrcAccessMask(vk::AccessFlagBits::eShaderRead);
  present_to_storage.setImage(colormap_img.img);
  present_to_storage.setSubresourceRange(
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  reduction_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   {}, nullptr, nullptr, present_to_storage);
  u32 X = (nx * ny) / 32;
  appendOp(reduction_buffer, first_minmax_reduction, X, 1, 1);
  X = (X + 31) / 32;
  while (X > 1) {
    appendOp(reduction_buffer, minmax_reduction, X, 1, 1);
    X = (X + 31) / 32;
  }
  appendOp(reduction_buffer, minmax_reduction, 1, 1, 1);
  appendOp(reduction_buffer, fill_colormap_img, nx / 8, ny / 8, 1);
  vk::ImageMemoryBarrier storage_to_present{};
  storage_to_present.setOldLayout(vk::ImageLayout::eGeneral);
  storage_to_present.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
  storage_to_present.setSrcQueueFamilyIndex(manager.cQFI);
  storage_to_present.setDstQueueFamilyIndex(manager.cQFI);
  storage_to_present.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  storage_to_present.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
  storage_to_present.setImage(colormap_img.img);
  storage_to_present.setSubresourceRange(
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  reduction_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   {}, nullptr, nullptr, storage_to_present);
  reduction_buffer.end();
  oneTimeSubmit(manager.device, manager.commandPool, manager.queue,
                [&](vk::CommandBuffer b) {
                  b.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    {}, nullptr, nullptr, storage_to_present);
                });
  for (size_t i = 0; i < n_images; i++) {
    commandBuffers[i].begin(begin_info);
    record_drawing_commands(swapChainFrameBuffers[i], renderPass,
                            swapChainExtent, graphicsPipeline,
                            graphicsPipelineLayout, descriptorSet,
                            vertexBuffer.buffer, commandBuffers[i]);
    commandBuffers[i].end();
  }
}

void Renderer::cleanupSwapchain() {
  vk::Device dev = p_mgr->device;
  dev.destroyCommandPool(command_pool);
  for (auto& fb : swapChainFrameBuffers) {
    dev.destroyFramebuffer(fb);
  }
  for (auto& iv : swapChainImageViews) {
    dev.destroyImageView(iv);
  }
  dev.destroySwapchainKHR(swapchain);
}
void Renderer::recreateSwapchain() {
  p_mgr->device.waitIdle();
  capabilities =
      p_mgr->physicalDevice.getSurfaceCapabilitiesKHR(p_mgr->surface);
  if (capabilities.currentExtent.width ==
      std::numeric_limits<uint32_t>::max()) {
    s32 width = 0, height = 0;
    // GetWindowSize seems to just return 0 on linux for this version of SDL3,
    // hopefully we never encounter this branch.
    SDL_GetWindowSize(p_mgr->window, &width, &height);
    u32 uwidth = (u32)width;
    u32 uheight = (u32)height;
    swapChainExtent.width =
        std::clamp(uwidth, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.height);
    swapChainExtent.height =
        std::clamp(uheight, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

  } else {
    swapChainExtent.width = std::clamp(capabilities.currentExtent.width,
                                       capabilities.minImageExtent.width,
                                       capabilities.maxImageExtent.width);
    swapChainExtent.height = std::clamp(capabilities.currentExtent.height,
                                        capabilities.minImageExtent.height,
                                        capabilities.maxImageExtent.height);
  }
  vk::SwapchainKHR new_swapchain = create_swapchain(
      p_mgr->device, p_mgr->surface, capabilities, surface_format, present_mode,
      swapChainExtent, n_images, render_queue_indices, swapchain);
  cleanupSwapchain();
  swapchain = new_swapchain;
  swapChainImages = p_mgr->device.getSwapchainImagesKHR(swapchain);
  swapChainImageViews =
      create_image_views(p_mgr->device, surface_format, swapChainImages);

  vk::CommandPoolCreateInfo command_pool_info({}, render_queue_indices[0]);
  swapChainFrameBuffers = create_framebuffers(
      p_mgr->device, swapChainImageViews, renderPass, swapChainExtent);
  command_pool = p_mgr->device.createCommandPool(command_pool_info);
  vk::CommandBufferAllocateInfo cbAllocInfo(
      command_pool, vk::CommandBufferLevel::ePrimary, n_images);
  commandBuffers.resize(n_images);
  vkAllocateCommandBuffers(p_mgr->device,
                           pcast<VkCommandBufferAllocateInfo>(&cbAllocInfo),
                           pcast<VkCommandBuffer>(commandBuffers.data()));
  vk::CommandBufferBeginInfo begin_info{};
  for (u32 i = 0; i < n_images; i++) {
    commandBuffers[i].begin(begin_info);
    record_drawing_commands(swapChainFrameBuffers[i], renderPass,
                            swapChainExtent, graphicsPipeline,
                            graphicsPipelineLayout, descriptorSet,
                            vertexBuffer.buffer, commandBuffers[i]);
    commandBuffers[i].end();
  }
}

void Renderer::drawFrame() {
  vk::Device dev = p_mgr->device;
  if (dev.waitForFences(inFlightFences[currentFrame], vk::True, -1) !=
      vk::Result::eSuccess) {
    runtime_exc("Failed waiting for fences");
  };
  try {
    vk::ResultValue<u32> res = dev.acquireNextImageKHR(
        swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);
    if (res.result != vk::Result::eSuccess &&
        res.result != vk::Result::eSuboptimalKHR) {
      runtime_exc("Failed to acquire swap chain image!");
    }
    u32 imageIndex = res.value;
    if (imageInFlightFences[imageIndex] != nullptr) {
      auto result =
          dev.waitForFences(imageInFlightFences[imageIndex], vk::True, -1);
      if (result != vk::Result::eSuccess) {
        runtime_exc("Failed to wait on in flight image");
      }
    }
    imageInFlightFences[imageIndex] = inFlightFences[currentFrame];

    vk::Semaphore signal_semaphore = renderFinishedSemaphores[currentFrame];
    vk::PipelineStageFlags waitDestinationStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::CommandBuffer cB = commandBuffers[imageIndex];
    vk::SubmitInfo submitInfo(imageAvailableSemaphores[currentFrame],
                              waitDestinationStageMask, cB, signal_semaphore);
    dev.resetFences(inFlightFences[currentFrame]);
    p_mgr->execute(reduction_buffer);

    graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

    vk::PresentInfoKHR pres_info(signal_semaphore, swapchain, imageIndex);
    try {
      if (vk::Result::eSuboptimalKHR == presentQueue.presentKHR(pres_info)) {
        std::cout << "Recreating swapchain because it is suboptimal\n";
        recreateSwapchain();
      }
    } catch (vk::OutOfDateKHRError& out_of_date_error) {
      recreateSwapchain();
      return;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  } catch (vk::OutOfDateKHRError& error) {
    recreateSwapchain();
    return;
  };
}

Renderer::~Renderer() {
  vk::Device dev = p_mgr->device;
  dev.waitIdle();
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    dev.destroySemaphore(imageAvailableSemaphores[i]);
    dev.destroySemaphore(renderFinishedSemaphores[i]);
    dev.destroyFence(inFlightFences[i]);
  }

  for (size_t i = 0; i < n_images; i++) {
    dev.destroyFramebuffer(swapChainFrameBuffers[i]);
    dev.destroyImageView(swapChainImageViews[i]);
  }
  dev.destroyCommandPool(command_pool);
  dev.destroyDescriptorPool(descriptorPool);
  dev.destroyDescriptorSetLayout(descriptorSetLayout);
  dev.destroySampler(colormap_sampler);
  dev.destroyImageView(colormap_view);
  dev.destroySwapchainKHR(swapchain);
  dev.destroyPipeline(graphicsPipeline);
  dev.destroyPipelineLayout(graphicsPipelineLayout);
  dev.destroyRenderPass(renderPass);
}*/
