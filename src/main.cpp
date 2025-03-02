#include "hack.h"
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

using std::bit_cast;

struct PositionTextureVertex {
  float x;
  float y;
  float u;
  float v;
};

static const char* BasePath = SDL_GetBasePath();

const int MAX_FRAMES_IN_FLIGHT = 2;

struct Init {
  SDL_Window* window;
  vkb::Instance instance;
  vkb::InstanceDispatchTable inst_disp;
  vk::SurfaceKHR surface;
  vkb::Device device;
  vkb::DispatchTable disp;
  vkb::Swapchain swapchain;
};

struct RenderData {
  vk::Queue graphics_queue;
  vk::Queue present_queue;

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
};

SDL_Window* create_window_sdl(const char* window_name = "", u32 flags = 0) {
  SDL_Init(SDL_INIT_VIDEO);
  if (!SDL_Vulkan_LoadLibrary(nullptr)) {
    SDL_Log("Unable to load Vulkan library: %s", SDL_GetError());
  };
  SDL_Window* window =
      SDL_CreateWindow("Bleh", 640, 480, SDL_WINDOW_VULKAN | flags);
  if (!window) {
    SDL_Log("CreateWindow failed with error: %s", SDL_GetError());
  }
  return window;
}

vk::SurfaceKHR create_surface_sdl(VkInstance instance, SDL_Window* window,
                                  VkAllocationCallbacks* allocator = nullptr) {
  vk::SurfaceKHR surf;
  if (!SDL_Vulkan_CreateSurface(window, instance, nullptr,
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
                          .build();
  if (!instance_ret) {
    std::cout << instance_ret.error().message() << "\n";
    return -1;
  }
  init.instance = instance_ret.value();

  init.inst_disp = init.instance.make_table();

  init.surface = create_surface_sdl(init.instance, init.window);

  vkb::PhysicalDeviceSelector phys_device_selector(init.instance);
  auto phys_device_ret =
      phys_device_selector.set_surface(init.surface).select();
  if (!phys_device_ret) {
    std::cout << phys_device_ret.error().message() << "\n";
    return -1;
  }
  vkb::PhysicalDevice physical_device = phys_device_ret.value();

  vkb::DeviceBuilder device_builder{physical_device};
  auto device_ret = device_builder.build();
  if (!device_ret) {
    std::cout << device_ret.error().message() << "\n";
    return -1;
  }
  init.device = device_ret.value();

  init.disp = init.device.make_table();

  return 0;
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

  vk::PipelineVertexInputStateCreateInfo vertex_input_info(
      vk::PipelineVertexInputStateCreateFlags(), 0, 0);

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

int create_command_pool(Init& init, RenderData& data) {
  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex =
      init.device.get_queue_index(vkb::QueueType::graphics).value();

  if (init.disp.createCommandPool(
          &pool_info, nullptr, bit_cast<VkCommandPool*>(&data.command_pool)) !=
      VK_SUCCESS) {
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

  for (size_t i = 0; i < data.command_buffers.size(); i++) {
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (init.disp.beginCommandBuffer(data.command_buffers[i], &begin_info) !=
        VK_SUCCESS) {
      return -1; // failed to begin recording command buffer
    }

    vk::ClearValue clearColor{{1.0f, 1.0f, 1.0f, 1.0f}};
    vk::RenderPassBeginInfo render_pass_info(
        data.render_pass, data.framebuffers[i],
        vk::Rect2D(vk::Offset2D(0, 0), init.swapchain.extent), clearColor);

    vk::Viewport viewport(0.0f, 0.0f, (f32)init.swapchain.extent.width,
                          (f32)init.swapchain.extent.height, 0.0f, 1.0f);

    vk::Rect2D scissor(vk::Offset2D(0, 0), init.swapchain.extent);

    vk::CommandBuffer cB = data.command_buffers[i];
    cB.setViewport(0, viewport);
    cB.setScissor(0, scissor);
    cB.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);
    cB.bindPipeline(vk::PipelineBindPoint::eGraphics, data.graphics_pipeline);
    cB.draw(3, 1, 0, 0);
    cB.endRenderPass();
    cB.end();
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
  if (0 != create_command_pool(init, data))
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

  for (auto framebuffer : data.framebuffers) {
    init.disp.destroyFramebuffer(framebuffer, nullptr);
  }

  init.disp.destroyPipeline(data.graphics_pipeline, nullptr);
  init.disp.destroyPipelineLayout(data.pipeline_layout, nullptr);
  init.disp.destroyRenderPass(data.render_pass, nullptr);

  for (auto& image_view : data.swapchain_image_views) {
    vkDestroyImageView(init.device, image_view, nullptr);
  }

  vkb::destroy_swapchain(init.swapchain);
  vkb::destroy_device(init.device);
  vkb::destroy_surface(init.instance, init.surface);
  vkb::destroy_instance(init.instance);
  SDL_DestroyWindow(init.window);
  SDL_Vulkan_UnloadLibrary();
  SDL_Quit();
}
/*struct Init {
  SDL_Window* window;
  vkb::Instance instance;
  vkb::InstanceDispatchTable inst_disp;
  VkSurfaceKHR surface;
  vkb::Device device;
  vkb::DispatchTable disp;
  vkb::Swapchain swapchain;
};

struct RenderData {
  VkQueue graphics_queue;
  VkQueue present_queue;

  std::vector<VkImage> swapchain_images;
  std::vector<VkImageView> swapchain_image_views;
  std::vector<VkFramebuffer> framebuffers;

  VkRenderPass render_pass;
  VkPipelineLayout pipeline_layout;
  VkPipeline graphics_pipeline;

  VkCommandPool command_pool;
  std::vector<VkCommandBuffer> command_buffers;

  std::vector<VkSemaphore> available_semaphores;
  std::vector<VkSemaphore> finished_semaphore;
  std::vector<VkFence> in_flight_fences;
  std::vector<VkFence> image_in_flight;
  size_t current_frame = 0;
};

SDL_Window* create_window_sdl(const char* window_name = "", u32 flags = 0) {
  SDL_Init(SDL_INIT_VIDEO);
  if (!SDL_Vulkan_LoadLibrary(nullptr)) {
    SDL_Log("Unable to load Vulkan library: %s", SDL_GetError());
  };
  SDL_Window* window =
      SDL_CreateWindow("Bleh", 640, 480, SDL_WINDOW_VULKAN | flags);
  if (!window) {
    SDL_Log("CreateWindow failed with error: %s", SDL_GetError());
  }
  return window;
}
GLFWwindow* create_window_glfw(const char* window_name = "", bool resize =
true) { glfwInit(); glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); if (!resize)
glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    return glfwCreateWindow(1024, 1024, window_name, NULL, NULL);
}

void destroy_window_sdl(SDL_Window* window) {
  SDL_DestroyWindow(window);
  SDL_Vulkan_UnloadLibrary();
  SDL_Quit();
}

VkSurfaceKHR create_surface_sdl(VkInstance instance, SDL_Window* window,
                                VkAllocationCallbacks* allocator = nullptr) {
  vk::SurfaceKHR surf;
  if (!SDL_Vulkan_CreateSurface(window, instance, nullptr,
                                bit_cast<VkSurfaceKHR*>(&surf))) {
    SDL_Log("CreateSurface failed with error: %s", SDL_GetError());
  }
  return surf;
}

VkSurfaceKHR create_surface_glfw(VkInstance instance, GLFWwindow* window,
                                 VkAllocationCallbacks* allocator = nullptr) {
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkResult err = glfwCreateWindowSurface(instance, window, allocator, &surface);
  if (err) {
    const char* error_msg;
    int ret = glfwGetError(&error_msg);
    if (ret != 0) {
      std::cout << ret << " ";
      if (error_msg != nullptr)
        std::cout << error_msg;
      std::cout << "\n";
    }
    surface = VK_NULL_HANDLE;
  }
  return surface;
}

int device_initialization(Init& init) {
  init.window = create_window_sdl("Vulkan Triangle");

  vkb::InstanceBuilder instance_builder;
  auto instance_ret = instance_builder.use_default_debug_messenger()
                          .request_validation_layers()
                          .build();
  if (!instance_ret) {
    std::cout << instance_ret.error().message() << "\n";
    return -1;
  }
  init.instance = instance_ret.value();

  init.inst_disp = init.instance.make_table();

  init.surface = create_surface_sdl(init.instance, init.window);

  vkb::PhysicalDeviceSelector phys_device_selector(init.instance);
  auto phys_device_ret =
      phys_device_selector.set_surface(init.surface).select();
  if (!phys_device_ret) {
    std::cout << phys_device_ret.error().message() << "\n";
    return -1;
  }
  vkb::PhysicalDevice physical_device = phys_device_ret.value();

  vkb::DeviceBuilder device_builder{physical_device};
  auto device_ret = device_builder.build();
  if (!device_ret) {
    std::cout << device_ret.error().message() << "\n";
    return -1;
  }
  init.device = device_ret.value();

  init.disp = init.device.make_table();

  return 0;
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
  return 0;
}

int create_render_pass(Init& init, RenderData& data) {
  VkAttachmentDescription color_attachment = {};
  color_attachment.format = init.swapchain.image_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref = {};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  VkSubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  if (init.disp.createRenderPass(&render_pass_info, nullptr,
                                 &data.render_pass) != VK_SUCCESS) {
    std::cout << "failed to create render pass\n";
    return -1; // failed to create render pass!
  }
  return 0;
}

std::vector<char> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t file_size = (size_t)file.tellg();
  std::vector<char> buffer(file_size);

  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(file_size));

  file.close();

  return buffer;
}

VkShaderModule createShaderModule(Init& init, const std::vector<char>& code) {
  VkShaderModuleCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule shaderModule;
  if (init.disp.createShaderModule(&create_info, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE; // failed to create shader module
  }

  return shaderModule;
}

int create_graphics_pipeline(Init& init, RenderData& data) {
  auto vert_code =
      readFile(std::string(BasePath) + "Shaders/triangle.vert.spv");
  auto frag_code =
      readFile(std::string(BasePath) + "Shaders/triangle.frag.spv");

  VkShaderModule vert_module = createShaderModule(init, vert_code);
  VkShaderModule frag_module = createShaderModule(init, frag_code);
  if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
    std::cout << "failed to create shader module\n";
    return -1; // failed to create shader modules
  }

  VkPipelineShaderStageCreateInfo vert_stage_info = {};
  vert_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vert_stage_info.module = vert_module;
  vert_stage_info.pName = "main";

  VkPipelineShaderStageCreateInfo frag_stage_info = {};
  frag_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_stage_info.module = frag_module;
  frag_stage_info.pName = "main";

  VkPipelineShaderStageCreateInfo shader_stages[] = {vert_stage_info,
                                                     frag_stage_info};

  VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
  vertex_input_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input_info.vertexBindingDescriptionCount = 0;
  vertex_input_info.vertexAttributeDescriptionCount = 0;

  VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
  input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport = {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)init.swapchain.extent.width;
  viewport.height = (float)init.swapchain.extent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor = {};
  scissor.offset = {0, 0};
  scissor.extent = init.swapchain.extent;

  VkPipelineViewportStateCreateInfo viewport_state = {};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo color_blending = {};
  color_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &colorBlendAttachment;
  color_blending.blendConstants[0] = 0.0f;
  color_blending.blendConstants[1] = 0.0f;
  color_blending.blendConstants[2] = 0.0f;
  color_blending.blendConstants[3] = 0.0f;

  VkPipelineLayoutCreateInfo pipeline_layout_info = {};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 0;
  pipeline_layout_info.pushConstantRangeCount = 0;

  if (init.disp.createPipelineLayout(&pipeline_layout_info, nullptr,
                                     &data.pipeline_layout) != VK_SUCCESS) {
    std::cout << "failed to create pipeline layout\n";
    return -1; // failed to create pipeline layout
  }

  std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,
                                                VK_DYNAMIC_STATE_SCISSOR};

  VkPipelineDynamicStateCreateInfo dynamic_info = {};
  dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
  dynamic_info.pDynamicStates = dynamic_states.data();

  VkGraphicsPipelineCreateInfo pipeline_info = {};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = 2;
  pipeline_info.pStages = shader_stages;
  pipeline_info.pVertexInputState = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = &dynamic_info;
  pipeline_info.layout = data.pipeline_layout;
  pipeline_info.renderPass = data.render_pass;
  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

  if (init.disp.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipeline_info,
                                        nullptr, &data.graphics_pipeline) !=
      VK_SUCCESS) {
    std::cout << "failed to create pipline\n";
    return -1; // failed to create graphics pipeline
  }

  init.disp.destroyShaderModule(frag_module, nullptr);
  init.disp.destroyShaderModule(vert_module, nullptr);
  return 0;
}

int create_framebuffers(Init& init, RenderData& data) {
  data.swapchain_images = init.swapchain.get_images().value();
  data.swapchain_image_views = init.swapchain.get_image_views().value();

  data.framebuffers.resize(data.swapchain_image_views.size());

  for (size_t i = 0; i < data.swapchain_image_views.size(); i++) {
    VkImageView attachments[] = {data.swapchain_image_views[i]};

    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = data.render_pass;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = attachments;
    framebuffer_info.width = init.swapchain.extent.width;
    framebuffer_info.height = init.swapchain.extent.height;
    framebuffer_info.layers = 1;

    if (init.disp.createFramebuffer(&framebuffer_info, nullptr,
                                    &data.framebuffers[i]) != VK_SUCCESS) {
      return -1; // failed to create framebuffer
    }
  }
  return 0;
}

int create_command_pool(Init& init, RenderData& data) {
  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex =
      init.device.get_queue_index(vkb::QueueType::graphics).value();

  if (init.disp.createCommandPool(&pool_info, nullptr, &data.command_pool) !=
      VK_SUCCESS) {
    std::cout << "failed to create command pool\n";
    return -1; // failed to create command pool
  }
  return 0;
}

int create_command_buffers(Init& init, RenderData& data) {
  data.command_buffers.resize(data.framebuffers.size());

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = data.command_pool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = (uint32_t)data.command_buffers.size();

  if (init.disp.allocateCommandBuffers(
          &allocInfo, data.command_buffers.data()) != VK_SUCCESS) {
    return -1; // failed to allocate command buffers;
  }

  for (size_t i = 0; i < data.command_buffers.size(); i++) {
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (init.disp.beginCommandBuffer(data.command_buffers[i], &begin_info) !=
        VK_SUCCESS) {
      return -1; // failed to begin recording command buffer
    }

    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = data.render_pass;
    render_pass_info.framebuffer = data.framebuffers[i];
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = init.swapchain.extent;
    VkClearValue clearColor{{{0.0f, 0.0f, 0.0f, 1.0f}}};
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clearColor;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)init.swapchain.extent.width;
    viewport.height = (float)init.swapchain.extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = init.swapchain.extent;

    init.disp.cmdSetViewport(data.command_buffers[i], 0, 1, &viewport);
    init.disp.cmdSetScissor(data.command_buffers[i], 0, 1, &scissor);

    init.disp.cmdBeginRenderPass(data.command_buffers[i], &render_pass_info,
                                 VK_SUBPASS_CONTENTS_INLINE);

    init.disp.cmdBindPipeline(data.command_buffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              data.graphics_pipeline);

    init.disp.cmdDraw(data.command_buffers[i], 3, 1, 0, 0);

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

  VkSemaphoreCreateInfo semaphore_info = {};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (init.disp.createSemaphore(&semaphore_info, nullptr,
                                  &data.available_semaphores[i]) !=
            VK_SUCCESS ||
        init.disp.createSemaphore(&semaphore_info, nullptr,
                                  &data.finished_semaphore[i]) != VK_SUCCESS ||
        init.disp.createFence(&fence_info, nullptr,
                              &data.in_flight_fences[i]) != VK_SUCCESS) {
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

  init.swapchain.destroy_image_views(data.swapchain_image_views);

  if (0 != create_swapchain(init))
    return -1;
  if (0 != create_framebuffers(init, data))
    return -1;
  if (0 != create_command_pool(init, data))
    return -1;
  if (0 != create_command_buffers(init, data))
    return -1;
  return 0;
}

int draw_frame(Init& init, RenderData& data) {
  init.disp.waitForFences(1, &data.in_flight_fences[data.current_frame],
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
    init.disp.waitForFences(1, &data.image_in_flight[image_index], VK_TRUE,
                            UINT64_MAX);
  }
  data.image_in_flight[image_index] = data.in_flight_fences[data.current_frame];

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore wait_semaphores[] = {
      data.available_semaphores[data.current_frame]};
  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = wait_semaphores;
  submitInfo.pWaitDstStageMask = wait_stages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &data.command_buffers[image_index];

  VkSemaphore signal_semaphores[] = {
      data.finished_semaphore[data.current_frame]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signal_semaphores;

  init.disp.resetFences(1, &data.in_flight_fences[data.current_frame]);

  if (init.disp.queueSubmit(data.graphics_queue, 1, &submitInfo,
                            data.in_flight_fences[data.current_frame]) !=
      VK_SUCCESS) {
    std::cout << "failed to submit draw command buffer\n";
    return -1; //"failed to submit draw command buffer
  }

  VkPresentInfoKHR present_info = {};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;

  VkSwapchainKHR swapChains[] = {init.swapchain};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapChains;

  present_info.pImageIndices = &image_index;

  result = init.disp.queuePresentKHR(data.present_queue, &present_info);
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

  for (auto framebuffer : data.framebuffers) {
    init.disp.destroyFramebuffer(framebuffer, nullptr);
  }

  init.disp.destroyPipeline(data.graphics_pipeline, nullptr);
  init.disp.destroyPipelineLayout(data.pipeline_layout, nullptr);
  init.disp.destroyRenderPass(data.render_pass, nullptr);

  init.swapchain.destroy_image_views(data.swapchain_image_views);

  vkb::destroy_swapchain(init.swapchain);
  vkb::destroy_device(init.device);
  vkb::destroy_surface(init.instance, init.surface);
  vkb::destroy_instance(init.instance);
  destroy_window_sdl(init.window);
}*/

int main(int argc, char* argv[]) {
  Init init;
  RenderData render_data;
  if (0 != device_initialization(init))
    return -1;
  if (0 != create_swapchain(init))
    return -1;
  if (0 != get_queues(init, render_data))
    return -1;
  if (0 != create_render_pass(init, render_data))
    return -1;
  if (0 != create_graphics_pipeline(init, render_data))
    return -1;
  if (0 != create_framebuffers(init, render_data))
    return -1;
  if (0 != create_command_pool(init, render_data))
    return -1;
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
