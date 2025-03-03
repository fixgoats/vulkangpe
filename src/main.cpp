#include "hack.h"
#include "vk_mem_alloc.h"
#include <cmath>
#include <cstring>
#include <cxxopts.hpp>
#include <iostream>

#include "SDL3/SDL.h"
#include "SDL3/SDL_vulkan.h"
#include "colormaps.h"
#include "mathhelpers.h"
#include "typedefs.h"
#include "vkcore.h"
#include <vulkan/vulkan.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

using std::bit_cast;

template <typename T, typename U, typename = void>
struct is_safely_castable : std::false_type {};

template <typename T, typename U>
struct is_safely_castable<
    T, U, std::void_t<decltype(static_cast<U>(std::declval<T>()))>>
    : std::true_type {};

template <class T, class B>
T* pcast(B* x) {
  static_assert(is_safely_castable<T, B>(), "Types are not equivalent");
  return bit_cast<T*>(x);
}

struct PositionTextureVertex {
  vec2<f32> pos;
  vec2<f32> uv;

  static vk::VertexInputBindingDescription bindingDscr() {
    return {0, sizeof(PositionTextureVertex), vk::VertexInputRate::eVertex};
  }
  static std::array<vk::VertexInputAttributeDescription, 2> attributeDscr() {
    return {{{0, 0, vk::Format::eR32G32Sfloat,
              offsetof(PositionTextureVertex, pos)},
             {1, 0, vk::Format::eR32G32Sfloat,
              offsetof(PositionTextureVertex, uv)}}};
  }
};

static const char* BasePath = SDL_GetBasePath();

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

struct Init {
  SDL_Window* window;
  vkb::Instance instance;
  vkb::InstanceDispatchTable inst_disp;
  vk::SurfaceKHR surface;
  vkb::Device device;
  vkb::DispatchTable disp;
  vkb::Swapchain swapchain;
  vk::CommandPool transfer_pool;
  vk::Queue transfer_queue;
  VmaAllocator allocator;
  MetaBuffer staging;
};

void copyToImage(Init& init, MetaBuffer& src, AllocatedImage& dst, u32 width,
                 u32 height) {
  auto commandBuffer =
      static_cast<vk::Device>(init.device)
          .allocateCommandBuffers(
              {init.transfer_pool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  vk::BufferImageCopy region{};
  region.setBufferOffset(0);
  region.setBufferRowLength(0);
  region.setBufferImageHeight(0);
  region.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
  region.setImageOffset({0, 0, 0});
  region.setImageExtent({width, height, 1});
  commandBuffer.copyBufferToImage(
      src.buffer, dst.img, vk::ImageLayout::eTransferDstOptimal, 1, &region);
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  init.transfer_queue.submit(submitInfo);
  init.transfer_queue.waitIdle();
  static_cast<vk::Device>(init.device)
      .freeCommandBuffers(init.transfer_pool, commandBuffer);
}

void writeToImage(Init& init, const void* src, AllocatedImage& dst,
                  vk::DeviceSize size, u32 width, u32 height) {
  if (size > init.staging.aInfo.size) {
    vmaDestroyBuffer(init.allocator, init.staging.buffer,
                     init.staging.allocation);
    vk::BufferCreateInfo stagingBCI({}, size,
                                    vk::BufferUsageFlagBits::eTransferSrc |
                                        vk::BufferUsageFlagBits::eTransferDst);
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    init.staging.allocate(init.allocator, allocCreateInfo, stagingBCI);
    /*init.staging.allocation = VmaAllocation{};
    init.staging.aInfo = VmaAllocationInfo{};
    vmaCreateBuffer(init.allocator,
    bit_cast<VkBufferCreateInfo*>(&stagingBCI), &allocCreateInfo,
    bit_cast<VkBuffer*>(&init.staging), &stagingAllocation, &stagingInfo);*/
  }

  memcpy(init.staging.aInfo.pMappedData, src, size);
  copyToImage(init, init.staging, dst, width, height);
}

struct RenderData {
  vk::Queue graphics_queue;
  vk::Queue present_queue;
  vk::Queue compute_queue;

  std::vector<vk::Image> swapchain_images;
  std::vector<vk::ImageView> swapchain_image_views;
  std::vector<vk::Framebuffer> framebuffers;

  vk::RenderPass render_pass;
  vk::PipelineLayout pipeline_layout;
  vk::Pipeline graphics_pipeline;

  vk::CommandPool command_pool;
  std::vector<vk::CommandBuffer> command_buffers;

  std::vector<vk::Semaphore> available_semaphores;
  std::vector<vk::Semaphore> finished_semaphore;
  std::vector<vk::Fence> in_flight_fences;
  std::vector<vk::Fence> image_in_flight;
  size_t current_frame = 0;

  MetaBuffer vertex_buffer;

  std::vector<vk::DescriptorSet> descriptor_set;
  vk::DescriptorPool descriptor_pool;
  vk::DescriptorSetLayout descriptor_set_layout;

  AllocatedImage colormap;
  vk::ImageView colormap_image_view;
  vk::Sampler colormap_sampler;
};

SDL_Window* create_window_sdl(const char* window_name = "", u32 flags = 0) {
  SDL_Init(SDL_INIT_VIDEO);
  if (!SDL_Vulkan_LoadLibrary(nullptr)) {
    SDL_Log("Unable to load Vulkan library: %s", SDL_GetError());
  };
  SDL_Window* window =
      SDL_CreateWindow(window_name, 640, 480, SDL_WINDOW_VULKAN | flags);
  if (!window) {
    SDL_Log("CreateWindow failed with error: %s", SDL_GetError());
  }
  return window;
}

vk::SurfaceKHR create_surface_sdl(VkInstance instance, SDL_Window* window,
                                  VkAllocationCallbacks* allocator = nullptr) {
  vk::SurfaceKHR surf;
  if (!SDL_Vulkan_CreateSurface(window, instance, allocator,
                                bit_cast<VkSurfaceKHR*>(&surf))) {
    SDL_Log("CreateSurface failed with error: %s", SDL_GetError());
  }
  return surf;
}

int device_initialization(Init& init) {
  init.window = create_window_sdl("Vulkan Triangle", true);

  vkb::InstanceBuilder instance_builder;
  auto instance_ret = instance_builder.use_default_debug_messenger()
                          .request_validation_layers()
                          .require_api_version(1, 3, 0)
                          .build();
  if (!instance_ret) {
    std::cout << instance_ret.error().message() << "\n";
    return -1;
  }
  init.instance = instance_ret.value();

  init.inst_disp = init.instance.make_table();

  init.surface = create_surface_sdl(init.instance, init.window);

  VkPhysicalDeviceVulkan13Features features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  features.synchronization2 = true;

  VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  features12.bufferDeviceAddress = true;
  features12.descriptorIndexing = true;
  features12.descriptorBindingPartiallyBound = true;
  features12.descriptorBindingVariableDescriptorCount = true;
  features12.runtimeDescriptorArray = true;

  vkb::PhysicalDeviceSelector phys_device_selector(init.instance);
  auto phys_device_ret = phys_device_selector.allow_any_gpu_device_type(false)
                             .set_minimum_version(1, 3)
                             .set_required_features_13(features)
                             .set_required_features_12(features12)
                             .set_surface(init.surface)
                             .select();
  if (!phys_device_ret) {
    std::cout << phys_device_ret.error().message() << "\n";
    return -1;
  }
  vkb::PhysicalDevice physical_device = phys_device_ret.value();
  std::cout << physical_device.name << '\n';

  vkb::DeviceBuilder device_builder{physical_device};
  auto device_ret = device_builder.build();
  if (!device_ret) {
    std::cout << device_ret.error().message() << "\n";
    return -1;
  }
  init.device = device_ret.value();

  init.disp = init.device.make_table();
  VmaAllocatorCreateInfo allocatorCreateInfo = {};
  allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  allocatorCreateInfo.physicalDevice = init.device.physical_device;
  allocatorCreateInfo.device = init.device;
  allocatorCreateInfo.instance = init.instance;
  vmaCreateAllocator(&allocatorCreateInfo, &init.allocator);

  vk::BufferCreateInfo stagingBCI({}, 1024 * 1024,
                                  vk::BufferUsageFlagBits::eTransferSrc);
  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  allocCreateInfo.priority = 1.0f;
  init.staging.allocate(init.allocator, allocCreateInfo, stagingBCI);
  return 0;
}

void create_image(Init& init, RenderData& data) {
  auto format = vk::Format::eR8G8B8A8Srgb;
  vk::ImageCreateInfo imageInfo{};
  imageInfo.setImageType(vk::ImageType::e2D);
  imageInfo.setExtent({640, 480, 1});
  imageInfo.setMipLevels(1);
  imageInfo.setArrayLayers(1);
  imageInfo.setFormat(format);
  imageInfo.setTiling(vk::ImageTiling::eOptimal);
  imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
  imageInfo.setUsage(vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eSampled |
                     vk::ImageUsageFlagBits::eColorAttachment);
  imageInfo.setSharingMode(vk::SharingMode::eExclusive);
  imageInfo.setSamples(vk::SampleCountFlagBits::e1);
  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocCreateInfo.priority = 1.0f;
  data.colormap.allocate(init.allocator, allocCreateInfo, imageInfo);
  s32 texWidth, texHeight, texChannels;
  stbi_uc* pixels = stbi_load("tex640x480.jpg", &texWidth, &texHeight,
                              &texChannels, STBI_rgb_alpha);
  vk::ImageMemoryBarrier barrier{};
  barrier.setOldLayout(vk::ImageLayout::eUndefined);
  barrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
  barrier.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
  barrier.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
  barrier.setImage(data.colormap.img);
  barrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  barrier.setSrcAccessMask({});
  barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  oneTimeSubmit(init.device.device, data.command_pool, data.graphics_queue,
                [&](vk::CommandBuffer b) {
                  b.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                    vk::PipelineStageFlagBits::eTransfer, {},
                                    nullptr, nullptr, barrier);
                });
  writeToImage(init, pixels, data.colormap, texWidth * texHeight * 4, texWidth,
               texHeight);
  barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
  barrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
  barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
  barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  oneTimeSubmit(init.device.device, data.command_pool, data.graphics_queue,
                [&](vk::CommandBuffer b) {
                  b.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eFragmentShader,
                                    {}, nullptr, nullptr, barrier);
                });

  vk::ImageViewCreateInfo viewInfo{};
  viewInfo.setImage(data.colormap.img);
  viewInfo.setViewType(vk::ImageViewType::e2D);
  viewInfo.setFormat(format);
  viewInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

  init.disp.createImageView(pcast<VkImageViewCreateInfo>(&viewInfo), nullptr,
                            pcast<VkImageView>(&data.colormap_image_view));

  vk::SamplerCreateInfo sampler_info{};
  sampler_info.setMagFilter(vk::Filter::eNearest);
  sampler_info.setMinFilter(vk::Filter::eNearest);
  sampler_info.setAddressModeU(vk::SamplerAddressMode::eRepeat);
  sampler_info.setAddressModeV(vk::SamplerAddressMode::eRepeat);
  sampler_info.setAddressModeW(vk::SamplerAddressMode::eRepeat);
  sampler_info.setBorderColor(vk::BorderColor::eFloatOpaqueBlack);
  sampler_info.setUnnormalizedCoordinates(false);
  sampler_info.setCompareEnable(false);
  sampler_info.setCompareOp(vk::CompareOp::eAlways);
  sampler_info.setMipmapMode(vk::SamplerMipmapMode::eNearest);
  sampler_info.setMipLodBias(0.0f);
  sampler_info.setMinLod(0.0f);
  sampler_info.setMaxLod(0.0f);

  init.disp.createSampler(pcast<VkSamplerCreateInfo>(&sampler_info), nullptr,
                          pcast<VkSampler>(&data.colormap_sampler));
}

void create_descriptor(Init& init, RenderData& data) {
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorCount = 1;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binding.pImmutableSamplers = nullptr;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo dsl_info = {};
  dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsl_info.bindingCount = 1;
  dsl_info.pBindings = &binding;
  init.disp.createDescriptorSetLayout(
      &dsl_info, nullptr,
      pcast<VkDescriptorSetLayout>(&data.descriptor_set_layout));

  std::vector<vk::DescriptorSetLayout> layouts(
      init.swapchain.get_image_views().value().size(),
      data.descriptor_set_layout);

  VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    MAX_FRAMES_IN_FLIGHT};

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = layouts.size();
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  init.disp.createDescriptorPool(
      &pool_info, nullptr, pcast<VkDescriptorPool>(&data.descriptor_pool));

  VkDescriptorSetAllocateInfo ds_allocate_info = {};
  data.descriptor_set.resize(layouts.size());
  ds_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ds_allocate_info.descriptorPool = data.descriptor_pool;
  ds_allocate_info.descriptorSetCount = layouts.size();
  ds_allocate_info.pSetLayouts = pcast<VkDescriptorSetLayout>(layouts.data());
  if (init.disp.allocateDescriptorSets(
          &ds_allocate_info,
          pcast<VkDescriptorSet>(data.descriptor_set.data())) != VK_SUCCESS) {
    std::cerr << "Failed to allocate descriptor sets";
  }

  for (auto& set : data.descriptor_set) {
    vk::DescriptorImageInfo img_info{};
    img_info.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    img_info.setImageView(data.colormap_image_view);
    img_info.setSampler(data.colormap_sampler);

    vk::WriteDescriptorSet writes{};
    writes.setDstSet(set);
    writes.setDstBinding(0);
    writes.setDstArrayElement(0);
    writes.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
    writes.setDescriptorCount(1);
    writes.setPImageInfo(&img_info);
    init.disp.updateDescriptorSets(1, pcast<VkWriteDescriptorSet>(&writes), 0,
                                   nullptr);
  }
}

int create_swapchain(Init& init) {

  vkb::SwapchainBuilder swapchain_builder{init.device};
  auto swap_ret = swapchain_builder.set_old_swapchain(init.swapchain).build();
  if (!swap_ret) {
    std::cout << swap_ret.error().message() << " " << swap_ret.vk_result()
              << "\n";
    return -1;
  }
  vkb::destroy_swapchain(init.swapchain);
  init.swapchain = swap_ret.value();
  return 0;
}

int get_queues(Init& init, RenderData& data) {
  auto gq = init.device.get_queue(vkb::QueueType::graphics);
  if (!gq.has_value()) {
    std::cout << "failed to get graphics queue: " << gq.error().message()
              << "\n";
    return -1;
  }
  data.graphics_queue = gq.value();

  auto pq = init.device.get_queue(vkb::QueueType::present);
  if (!pq.has_value()) {
    std::cout << "failed to get present queue: " << pq.error().message()
              << "\n";
    return -1;
  }
  data.present_queue = pq.value();

  auto tq = init.device.get_queue(vkb::QueueType::transfer);
  if (!tq.has_value()) {
    std::cout << "failed to get transfer queue: " << tq.error().message()
              << "\n";
    return -1;
  }
  init.transfer_queue = tq.value();
  return 0;
}

int create_render_pass(Init& init, RenderData& data) {
  vk::AttachmentDescription color_attachment(
      vk::AttachmentDescriptionFlags(),
      static_cast<vk::Format>(init.swapchain.image_format),
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

  if (init.disp.createRenderPass(
          bit_cast<VkRenderPassCreateInfo*>(&render_pass_info), nullptr,
          bit_cast<VkRenderPass*>(&data.render_pass)) != VK_SUCCESS) {
    std::cout << "failed to create render pass\n";
    return -1; // failed to create render pass!
  }
  return 0;
}

vk::ShaderModule createShaderModule(Init& init, const std::vector<u32>& code) {
  vk::ShaderModuleCreateInfo create_info(vk::ShaderModuleCreateFlags(), code);

  VkShaderModule shaderModule;
  if (init.disp.createShaderModule(
          bit_cast<VkShaderModuleCreateInfo*>(&create_info), nullptr,
          &shaderModule) != VK_SUCCESS) {
    return VK_NULL_HANDLE; // failed to create shader module
  }

  return static_cast<vk::ShaderModule>(shaderModule);
}

int create_graphics_pipeline(Init& init, RenderData& data) {
  std::cout << "reading vertex shader code at " << BasePath
            << "Shaders/triangle.vert.spv\n";
  auto vert_code =
      readFile<u32>(std::string(BasePath) + "Shaders/triangle.vert.spv");
  std::cout << "reading fragment shader code at " << BasePath
            << "Shaders/triangle.frag.spv\n";
  auto frag_code =
      readFile<u32>(std::string(BasePath) + "Shaders/triangle.frag.spv");

  vk::ShaderModule vert_module = createShaderModule(init, vert_code);
  vk::ShaderModule frag_module = createShaderModule(init, frag_code);
  if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
    std::cout << "failed to create shader module\n";
    return -1; // failed to create shader modules
  }

  vk::PipelineShaderStageCreateInfo vert_stage_info(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
      vert_module, "main");

  vk::PipelineShaderStageCreateInfo frag_stage_info(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment,
      frag_module, "main");

  vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_stage_info,
                                                       frag_stage_info};

  auto binding = PositionTextureVertex::bindingDscr();
  auto attr = PositionTextureVertex::attributeDscr();
  vk::PipelineVertexInputStateCreateInfo vertex_input_info(
      vk::PipelineVertexInputStateCreateFlags(), binding, attr);

  vk::PipelineInputAssemblyStateCreateInfo input_assembly(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::eTriangleList);
  vk::Viewport viewport(0.0, 0.0, (f32)init.swapchain.extent.width,
                        (f32)init.swapchain.extent.height, 0.0f, 1.0f);

  vk::Rect2D scissor(vk::Offset2D(0, 0), init.swapchain.extent);

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

  vk::PipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.setSetLayouts(data.descriptor_set_layout);

  if (init.disp.createPipelineLayout(
          bit_cast<VkPipelineLayoutCreateInfo*>(&pipeline_layout_info), nullptr,
          bit_cast<VkPipelineLayout*>(&data.pipeline_layout)) != VK_SUCCESS) {
    std::cout << "failed to create pipeline layout\n";
    return -1; // failed to create pipeline layout
  }

  std::vector<vk::DynamicState> dynamic_states = {vk::DynamicState::eViewport,
                                                  vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamic_info(
      vk::PipelineDynamicStateCreateFlags(), dynamic_states);

  vk::GraphicsPipelineCreateInfo pipeline_info(
      vk::PipelineCreateFlags(), shader_stages, &vertex_input_info,
      &input_assembly, nullptr, &viewport_state, &rasterizer, &multisampling,
      nullptr, &color_blending, &dynamic_info, data.pipeline_layout,
      data.render_pass);

  if (init.disp.createGraphicsPipelines(
          VK_NULL_HANDLE, 1,
          bit_cast<VkGraphicsPipelineCreateInfo*>(&pipeline_info), nullptr,
          bit_cast<VkPipeline*>(&data.graphics_pipeline)) != VK_SUCCESS) {
    std::cout << "failed to create pipline\n";
    return -1; // failed to create graphics pipeline
  }

  init.disp.destroyShaderModule(frag_module, nullptr);
  init.disp.destroyShaderModule(vert_module, nullptr);
  return 0;
}

int create_framebuffers(Init& init, RenderData& data) {
  data.swapchain_images = [](Init& init) {
    auto tmp = init.swapchain.get_images().value();
    std::vector<vk::Image> ret(tmp.size());
    std::transform(tmp.cbegin(), tmp.cend(), ret.begin(),
                   [](VkImage x) { return static_cast<vk::Image>(x); });
    return ret;
  }(init);
  data.swapchain_image_views = [](Init& init) {
    auto tmp = init.swapchain.get_image_views().value();
    std::vector<vk::ImageView> ret(tmp.size());
    std::transform(tmp.cbegin(), tmp.cend(), ret.begin(),
                   [](VkImageView x) { return static_cast<vk::ImageView>(x); });
    return ret;
  }(init);

  data.framebuffers.resize(data.swapchain_image_views.size());

  for (size_t i = 0; i < data.swapchain_image_views.size(); i++) {
    vk::ImageView attachments[] = {data.swapchain_image_views[i]};

    vk::FramebufferCreateInfo framebuffer_info(
        {}, data.render_pass, attachments, init.swapchain.extent.width,
        init.swapchain.extent.height, 1);

    if (init.disp.createFramebuffer(
            bit_cast<VkFramebufferCreateInfo*>(&framebuffer_info), nullptr,
            bit_cast<VkFramebuffer*>(&data.framebuffers[i])) != VK_SUCCESS) {
      return -1; // failed to create framebuffer
    }
  }
  return 0;
}

int create_command_pool(Init& init, VkCommandPool* pool,
                        vkb::QueueType queue_type) {
  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = init.device.get_queue_index(queue_type).value();

  if (init.disp.createCommandPool(&pool_info, nullptr, pool) != VK_SUCCESS) {
    std::cout << "failed to create command pool\n";
    return -1; // failed to create command pool
  }
  return 0;
}

int create_command_buffers(Init& init, RenderData& data) {
  data.command_buffers.resize(data.framebuffers.size());

  vk::CommandBufferAllocateInfo allocInfo(data.command_pool,
                                          vk::CommandBufferLevel::ePrimary,
                                          (u32)data.command_buffers.size());

  if (init.disp.allocateCommandBuffers(
          bit_cast<VkCommandBufferAllocateInfo*>(&allocInfo),
          bit_cast<VkCommandBuffer*>(data.command_buffers.data())) !=
      VK_SUCCESS) {
    return -1; // failed to allocate command buffers;
  }

  std::cout << data.command_buffers.size() << '\n';
  for (size_t i = 0; i < data.command_buffers.size(); i++) {
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (init.disp.beginCommandBuffer(data.command_buffers[i], &begin_info) !=
        VK_SUCCESS) {
      return -1; // failed to begin recording command buffer
    }
    std::cout << data.descriptor_set[i] << '\n';

    vk::ClearValue clearColor{{1.0f, 1.0f, 1.0f, 1.0f}};
    vk::RenderPassBeginInfo render_pass_info(
        data.render_pass, data.framebuffers[i],
        vk::Rect2D(vk::Offset2D(0, 0), init.swapchain.extent), clearColor);

    vk::Viewport viewport(0.0f, 0.0f, (f32)init.swapchain.extent.width,
                          (f32)init.swapchain.extent.height, 0.0f, 1.0f);

    vk::Rect2D scissor(vk::Offset2D(0, 0), init.swapchain.extent);

    init.disp.cmdSetViewport(data.command_buffers[i], 0, 1,
                             pcast<VkViewport>(&viewport));
    init.disp.cmdSetScissor(data.command_buffers[i], 0, 1,
                            pcast<VkRect2D>(&scissor));
    init.disp.cmdBeginRenderPass(
        data.command_buffers[i],
        pcast<VkRenderPassBeginInfo>(&render_pass_info),
        VK_SUBPASS_CONTENTS_INLINE);
    init.disp.cmdBindPipeline(data.command_buffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              data.graphics_pipeline);
    init.disp.cmdBindDescriptorSets(
        data.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
        data.pipeline_layout, 0, 1,
        pcast<VkDescriptorSet>(&data.descriptor_set[i]), 0, nullptr);
    VkDeviceSize offsets[] = {0};
    init.disp.cmdBindVertexBuffers(data.command_buffers[i], 0, 1,
                                   pcast<VkBuffer>(&data.vertex_buffer.buffer),
                                   offsets);
    init.disp.cmdDraw(data.command_buffers[i], 6, 1, 0, 0);
    init.disp.cmdEndRenderPass(data.command_buffers[i]);
    if (init.disp.endCommandBuffer(data.command_buffers[i]) != VK_SUCCESS) {
      std::cout << "failed to record command buffer\n";
      return -1; // failed to record command buffer!
    }
  }
  return 0;
}

int create_sync_objects(Init& init, RenderData& data) {
  data.available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  data.finished_semaphore.resize(MAX_FRAMES_IN_FLIGHT);
  data.in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
  data.image_in_flight.resize(init.swapchain.image_count, VK_NULL_HANDLE);

  vk::SemaphoreCreateInfo semaphore_info{};

  vk::FenceCreateInfo fence_info(vk::FenceCreateFlagBits::eSignaled);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (init.disp.createSemaphore(
            bit_cast<VkSemaphoreCreateInfo*>(&semaphore_info), nullptr,
            bit_cast<VkSemaphore*>(&data.available_semaphores[i])) !=
            VK_SUCCESS ||
        init.disp.createSemaphore(
            bit_cast<VkSemaphoreCreateInfo*>(&semaphore_info), nullptr,
            bit_cast<VkSemaphore*>(&data.finished_semaphore[i])) !=
            VK_SUCCESS ||
        init.disp.createFence(
            bit_cast<VkFenceCreateInfo*>(&fence_info), nullptr,
            bit_cast<VkFence*>(&data.in_flight_fences[i])) != VK_SUCCESS) {
      std::cout << "failed to create sync objects\n";
      return -1; // failed to create synchronization objects for a frame
    }
  }
  return 0;
}

int recreate_swapchain(Init& init, RenderData& data) {
  init.disp.deviceWaitIdle();

  init.disp.destroyCommandPool(data.command_pool, nullptr);

  for (auto framebuffer : data.framebuffers) {
    init.disp.destroyFramebuffer(framebuffer, nullptr);
  }

  for (auto& image_view : data.swapchain_image_views) {
    vkDestroyImageView(init.device, image_view, nullptr);
  }

  if (0 != create_swapchain(init))
    return -1;
  if (0 != create_framebuffers(init, data))
    return -1;
  if (0 != create_command_pool(init,
                               bit_cast<VkCommandPool*>(&data.command_pool),
                               vkb::QueueType::graphics))
    return -1;
  if (0 != create_command_buffers(init, data))
    return -1;
  return 0;
}

int draw_frame(Init& init, RenderData& data) {
  init.disp.waitForFences(
      1, bit_cast<VkFence*>(&data.in_flight_fences[data.current_frame]),
      VK_TRUE, UINT64_MAX);

  uint32_t image_index = 0;
  VkResult result = init.disp.acquireNextImageKHR(
      init.swapchain, UINT64_MAX, data.available_semaphores[data.current_frame],
      VK_NULL_HANDLE, &image_index);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    return recreate_swapchain(init, data);
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    std::cout << "failed to acquire swapchain image. Error " << result << "\n";
    return -1;
  }

  if (data.image_in_flight[image_index] != VK_NULL_HANDLE) {
    init.disp.waitForFences(
        1, bit_cast<VkFence*>(&data.image_in_flight[image_index]), VK_TRUE,
        UINT64_MAX);
  }
  data.image_in_flight[image_index] = data.in_flight_fences[data.current_frame];

  vk::Semaphore signal_semaphore = data.finished_semaphore[data.current_frame];
  vk::PipelineStageFlags stage_flags(
      vk::PipelineStageFlagBits::eColorAttachmentOutput);
  vk::SubmitInfo submitInfo(1, &(data.available_semaphores[data.current_frame]),
                            &stage_flags, 1, &data.command_buffers[image_index],
                            1, &(signal_semaphore));

  init.disp.resetFences(
      1, bit_cast<VkFence*>(&data.in_flight_fences[data.current_frame]));

  if (init.disp.queueSubmit(
          data.graphics_queue, 1, bit_cast<VkSubmitInfo*>(&submitInfo),
          data.in_flight_fences[data.current_frame]) != VK_SUCCESS) {
    std::cout << "failed to submit draw command buffer\n";
    return -1; //"failed to submit draw command buffer
  }

  vk::SwapchainKHR b = static_cast<VkSwapchainKHR>(init.swapchain);
  vk::PresentInfoKHR present_info(signal_semaphore, b, image_index);

  result = init.disp.queuePresentKHR(
      data.present_queue, bit_cast<VkPresentInfoKHR*>(&present_info));
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    return recreate_swapchain(init, data);
  } else if (result != VK_SUCCESS) {
    std::cout << "failed to present swapchain image\n";
    return -1;
  }

  data.current_frame = (data.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
  return 0;
}

void cleanup(Init& init, RenderData& data) {
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    init.disp.destroySemaphore(data.finished_semaphore[i], nullptr);
    init.disp.destroySemaphore(data.available_semaphores[i], nullptr);
    init.disp.destroyFence(data.in_flight_fences[i], nullptr);
  }

  init.disp.destroyCommandPool(data.command_pool, nullptr);
  init.disp.destroyCommandPool(init.transfer_pool, nullptr);
  init.disp.destroyDescriptorSetLayout(data.descriptor_set_layout, nullptr);
  init.disp.destroyDescriptorPool(data.descriptor_pool, nullptr);

  for (auto framebuffer : data.framebuffers) {
    init.disp.destroyFramebuffer(framebuffer, nullptr);
  }

  init.disp.destroyPipeline(data.graphics_pipeline, nullptr);
  init.disp.destroyPipelineLayout(data.pipeline_layout, nullptr);
  init.disp.destroyRenderPass(data.render_pass, nullptr);

  for (auto& image_view : data.swapchain_image_views) {
    init.disp.destroyImageView(static_cast<VkImageView>(image_view), nullptr);
  }
  init.disp.destroySampler(data.colormap_sampler, nullptr);
  init.disp.destroyImageView(data.colormap_image_view, nullptr);
  vmaDestroyImage(init.allocator, data.colormap.img, data.colormap.allocation);

  vmaDestroyBuffer(init.allocator, init.staging.buffer,
                   init.staging.allocation);
  vmaDestroyBuffer(init.allocator, data.vertex_buffer.buffer,
                   data.vertex_buffer.allocation);
  vmaDestroyAllocator(init.allocator);
  vkb::destroy_swapchain(init.swapchain);
  vkb::destroy_device(init.device);
  vkb::destroy_surface(init.instance, init.surface);
  vkb::destroy_instance(init.instance);
  SDL_DestroyWindow(init.window);
  SDL_Vulkan_UnloadLibrary();
  SDL_Quit();
}

MetaBuffer make_staging_buffer(Init& init, size_t size) {
  vk::BufferCreateInfo stagingBCI({}, size,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst);
  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  allocCreateInfo.priority = 1.0f;
  return MetaBuffer(init.allocator, allocCreateInfo, stagingBCI);
}

void copyBuffer(Init& init, MetaBuffer& src, MetaBuffer& dst,
                u32 size = UINT32_MAX) {
  if (size == UINT32_MAX)
    size = src.aInfo.size;
  auto commandBuffer =
      static_cast<vk::Device>(init.device)
          .allocateCommandBuffers(
              {init.transfer_pool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  commandBuffer.copyBuffer(src.buffer, dst.buffer, vk::BufferCopy(0, 0, size));
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  init.transfer_queue.submit(submitInfo);
  init.transfer_queue.waitIdle();
  static_cast<vk::Device>(init.device)
      .freeCommandBuffers(init.transfer_pool, commandBuffer);
}

void writeToBuffer(Init& init, const void* src, MetaBuffer& dst, size_t size) {
  if (size > init.staging.aInfo.size) {
    vmaDestroyBuffer(init.allocator, init.staging.buffer,
                     init.staging.allocation);
    vk::BufferCreateInfo stagingBCI({}, size,
                                    vk::BufferUsageFlagBits::eTransferSrc |
                                        vk::BufferUsageFlagBits::eTransferDst);
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    init.staging.allocate(init.allocator, allocCreateInfo, stagingBCI);
    /*init.staging.allocation = VmaAllocation{};
    init.staging.aInfo = VmaAllocationInfo{};
    vmaCreateBuffer(init.allocator,
    bit_cast<VkBufferCreateInfo*>(&stagingBCI), &allocCreateInfo,
    bit_cast<VkBuffer*>(&init.staging), &stagingAllocation, &stagingInfo);*/
  }

  memcpy(init.staging.aInfo.pMappedData, src, size);
  copyBuffer(init, init.staging, dst, size);
}

template <class T>
void vecToBuffer(Init& init, std::vector<T> v, MetaBuffer& dst) {
  writeToBuffer(init, v.data(), dst, v.size() * sizeof(T));
}

void make_vertex_buffer(Init& init, RenderData& data,
                        const std::vector<PositionTextureVertex>& vertices) {
  vk::BufferCreateInfo vertexBCI(
      {}, vertices.size() * sizeof(PositionTextureVertex),
      vk::BufferUsageFlagBits::eVertexBuffer |
          vk::BufferUsageFlagBits::eTransferDst);
  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocCreateInfo.priority = 1.0f;
  data.vertex_buffer.allocate(init.allocator, allocCreateInfo, vertexBCI);
  vecToBuffer(init, vertices, data.vertex_buffer);
}

int main(int argc, char* argv[]) {
  Init init;
  RenderData render_data;
  if (0 != device_initialization(init))
    return -1;
  if (0 != create_swapchain(init))
    return -1;
  if (0 != get_queues(init, render_data))
    return -1;
  if (0 != create_command_pool(
               init, bit_cast<VkCommandPool*>(&render_data.command_pool),
               vkb::QueueType::graphics))
    return -1;
  if (0 != create_command_pool(init,
                               bit_cast<VkCommandPool*>(&init.transfer_pool),
                               vkb::QueueType::transfer))
    return -1;
  create_image(init, render_data);
  create_descriptor(init, render_data);
  if (0 != create_render_pass(init, render_data))
    return -1;
  if (0 != create_graphics_pipeline(init, render_data))
    return -1;
  if (0 != create_framebuffers(init, render_data))
    return -1;
  std::vector<PositionTextureVertex> vertices = {
      {{-1.0f, -1.0f}, {0.0f, 0.0f}}, {{1.0f, -1.0f}, {1.0f, 0.0f}},
      {{1.0f, 1.0f}, {1.0f, 1.0f}},   {{1.0f, 1.0f}, {1.0f, 1.0f}},
      {{-1.0f, 1.0f}, {0.0f, 1.0f}},  {{-1.0f, -1.0f}, {0.0f, 0.0f}},
  };
  make_vertex_buffer(init, render_data, vertices);
  if (0 != create_command_buffers(init, render_data))
    return -1;
  if (0 != create_sync_objects(init, render_data))
    return -1;

  bool running = true;
  while (running) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT ||
          (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
           event.window.windowID == SDL_GetWindowID(init.window))) {
        running = false;
      }
      if (SDL_GetWindowFlags(init.window) & SDL_WINDOW_MINIMIZED) {
        SDL_Delay(10);
        continue;
      }
      s32 res = draw_frame(init, render_data);
      if (res != 0) {
        std::cout << "Failed to draw frame\n";
        return -1;
      }
    }
  }
  init.disp.deviceWaitIdle();

  cleanup(init, render_data);
  return 0;
}
