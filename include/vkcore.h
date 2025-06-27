#pragma once
#include "SDL3/SDL.h"
#include "betterexc.h"
#include "hack.h"
#include "mathhelpers.h"
#include "metaprogramming.h"
#include "typedefs.h"
#include "vkFFT.h"
#include "vk_mem_alloc.h"
#include <boost/pfr/core.hpp>
#include <chrono>
#include <cstddef>
#include <format>
#include <fstream>
#include <span>

using std::bit_cast;

constexpr u32 GRID_WIDTH = 512;
constexpr u32 GRID_HEIGHT = 512;

constexpr u32 round_up_x16(u32 n) { return ((n + 15) / 16) * 16; }

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

template <typename T>
std::vector<T> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw runtime_exc("failed to open file: {}!", filename);
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<T> buffer(fileSize / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
  file.close();
  return buffer;
}

inline std::string tstamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t time = std::chrono::system_clock::to_time_t(now);
  tm local_tm = *localtime(&time);
  return std::format("{}-{}-{}/{}-{}", local_tm.tm_year - 100,
                     local_tm.tm_mon + 1, local_tm.tm_mday, local_tm.tm_hour,
                     local_tm.tm_min);
}

void saveToFile(std::string fname, const char* buf, size_t size);

constexpr u32 maxFramesInFlight = 2;

/*struct SimConstants {
  u32 nElementsX;
  u32 nElementsY;
  u32 nElementsZ;
  u32 xGroupSize;
  u32 yGroupSize;
  f32 gamma;
  f32 Gamma;
  f32 R;
  f32 EXY;
  f32 dt;
  f32 B0;
  f32 width;
  f32 resgamma;
  f32 Bamp;
  f32 PRatioMax;
  u32 substeps;
  constexpr u32 X() const { return nElementsX / xGroupSize; }
  constexpr u32 Y() const { return nElementsY / yGroupSize; }
  constexpr bool validate() const {
    return (nElementsY % yGroupSize == 0) && (nElementsX % xGroupSize == 0);
  }
  constexpr u32 elementsTotal() const { return nElementsX * nElementsY; }
};*/

// std::ostream& operator<<(std::ostream& os, const SimConstants& obj);
// std::ofstream& operator<<(std::ofstream& os, const SimConstants& obj);

const vk::MemoryBarrier fullMemoryBarrier(vk::AccessFlagBits::eMemoryWrite,
                                          vk::AccessFlagBits::eMemoryRead);

struct MetaBuffer {
  // A buffer + allocation stuff that you generally need to reference when using
  // vk::Buffers. Also destroys itself automatically.
  VmaAllocator* p_allocator = nullptr;
  vk::Buffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo aInfo;
  MetaBuffer();
  MetaBuffer(VmaAllocator& allocator, VmaAllocationCreateInfo& allocCreateInfo,
             vk::BufferCreateInfo& BCI);
  // To call on default constructed metabuffer
  void allocate(VmaAllocator& allocator,
                VmaAllocationCreateInfo& allocCreateInfo,
                vk::BufferCreateInfo& BCI);
  ~MetaBuffer();
};

struct AllocatedImage {
  VmaAllocator* p_allocator = nullptr;
  vk::Image img;
  VmaAllocation allocation;
  VmaAllocationInfo aInfo;
  AllocatedImage();
  AllocatedImage(VmaAllocator& allocator,
                 VmaAllocationCreateInfo& allocCreateInfo,
                 vk::ImageCreateInfo& iCI);
  void allocate(VmaAllocator& allocator,
                VmaAllocationCreateInfo& allocCreateInfo,
                vk::ImageCreateInfo& iCI);
  ~AllocatedImage();
};

struct SSBO430 {
  u32 n_fields;

  MetaBuffer b;
};

struct Algorithm {
  // Attention, device is used for destroying owned objects, the device will be
  // destroyed by the manager.
  vk::Device m_device;
  // owned
  vk::DescriptorSetLayout m_DSL;
  vk::DescriptorPool m_DescriptorPool;
  vk::DescriptorSet m_DescriptorSet;
  vk::ShaderModule m_ShaderModule;
  vk::PipelineLayout m_PipelineLayout;
  vk::Pipeline m_Pipeline;
  Algorithm() = default;
  Algorithm(vk::Device device, const std::vector<vk::ImageView>& img_views,
            const std::vector<MetaBuffer*>& buffers,
            const std::vector<u32>& spirv, const u8* specConsts = nullptr,
            const size_t* sizes = nullptr, size_t nConsts = 0,
            const size_t* pushSizes = nullptr, size_t nPushConstants = 0);
  void initialize(vk::Device device,
                  const std::vector<vk::ImageView>& img_views,
                  const std::vector<MetaBuffer*>& buffers,
                  const std::vector<u32>& spirv, const u8* specConsts = nullptr,
                  const size_t* sizes = nullptr, size_t nConsts = 0,
                  const size_t* pushSizes = nullptr, size_t nPushConstants = 0);
  ~Algorithm();
};

struct RaiiVkFFTApp {
  VkFFTApplication app;
  ~RaiiVkFFTApp() { deleteVkFFT(&app); }
};

struct RaiiVkFFTConf {
  std::vector<u64> bufferSizes;
  VkFFTConfiguration conf;
};

static const std::vector<std::string> deviceExtensions = {
    vk::KHRSwapchainExtensionName};

struct Manager {
  vk::Instance instance;
  vk::PhysicalDevice physicalDevice;
  vk::Device device;
  vk::Queue queue;
  vk::Fence fence;
  VmaAllocator allocator;
  vk::Buffer staging;
  VmaAllocation stagingAllocation;
  VmaAllocationInfo stagingInfo;
  u32 cQFI = UINT32_MAX;
  vk::CommandPool commandPool;
  SDL_Window* window;
  vk::SurfaceKHR surface;

  Manager(size_t stagingSize, SDL_Window* window = nullptr);
  void finishSetup(size_t stagingSize, vk::SurfaceKHR& surface);
  // Manager uses a single staging buffer for efficient copies.
  void copyBuffer(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer, u32 bufferSize,
                  u32 src_offset = 0, u32 dst_offset = 0);
  void copyInBatches(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                     u32 batchSize, u32 numBatches);
  vk::CommandBuffer copyOp(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                           u32 bufferSize, u32 src_offset = 0,
                           u32 dst_offset = 0);

  vk::CommandBuffer beginRecord(vk::CommandBufferUsageFlagBits bits = {});
  void execute(vk::CommandBuffer& b);
  void executeNoSync(vk::CommandBuffer& b);
  void queueWaitIdle();
  void getQueueFamilyIndices(vk::SurfaceKHR& surface);
  void writeToBuffer(MetaBuffer& buffer, const void* input, size_t size,
                     size_t src_offset = 0, size_t dst_offset = 0);
  template <class T>
  void writeToBuffer(MetaBuffer& buffer, std::vector<T> vec) {
    writeToBuffer(buffer, vec.data(), vec.size() * sizeof(T));
  }
  void writeFromBuffer(MetaBuffer& buffer, void* output, size_t size);
  template <class T>
  void writeFromBuffer(MetaBuffer& buffer, std::vector<T>& v) {
    writeFromBuffer(buffer, v.data(), v.size() * sizeof(T));
  }
  template <class T>
  void defaultInitBuffer(MetaBuffer& buffer, u32 nElements) {
    T* TStagingPtr = bit_cast<T*>(stagingInfo.pMappedData);
    for (u32 i = 0; i < nElements; i++) {
      TStagingPtr[i] = {};
    }
    copyBuffer(staging, buffer.buffer, nElements * sizeof(T));
  }
  template <typename T>
  MetaBuffer makeRawBuffer(u32 nElements) {
    vk::BufferCreateInfo bCI{vk::BufferCreateFlags(),
                             round_up_x16(nElements * sizeof(T)),
                             vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc,
                             vk::SharingMode::eExclusive,
                             1,
                             &cQFI};
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    allocCreateInfo.priority = 1.0f;
    return MetaBuffer{allocator, allocCreateInfo, bCI};
  }
  template <typename T>
  MetaBuffer makeUniformObject(T obj) {
    vk::BufferCreateInfo bCI{vk::BufferCreateFlags(),
                             sizeof(T),
                             vk::BufferUsageFlagBits::eUniformBuffer |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eTransferSrc,
                             vk::SharingMode::eExclusive,
                             1,
                             &cQFI};
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    allocCreateInfo.priority = 1.0f;
    return MetaBuffer{allocator, allocCreateInfo, bCI};
  }

  void freeCommandBuffer(vk::CommandBuffer& b) {
    device.freeCommandBuffers(commandPool, 1, &b);
  }

  void freeCommandBuffers(vk::CommandBuffer* b, u32 n) {
    device.freeCommandBuffers(commandPool, n, b);
  }

  template <typename T>
  MetaBuffer vecToBuffer(const std::vector<T>& v) {
    auto buffer = makeRawBuffer<T>(v.size());
    writeToBuffer(buffer, v);
    return buffer;
  }
  Algorithm makeAlgorithmRaw(
      std::string spirvname, const std::vector<vk::ImageView>& images,
      const std::vector<MetaBuffer*>& buffers, const u8* specConsts = nullptr,
      const size_t* specConstOffsets = nullptr, size_t nConsts = 0,
      const size_t* pushSizes = nullptr, size_t nPushConstants = 0);
  template <class T>
  Algorithm
  makeAlgorithm(std::string spirvname, const std::vector<vk::ImageView>& images,
                std::vector<MetaBuffer*> buffers, const T specConsts) {
    constexpr auto sizes = struct_field_sizes<T>();
    constexpr auto n_fields = sizes.size();
    return makeAlgorithmRaw(spirvname, images, buffers,
                            bit_cast<const u8*>(&specConsts), sizes.data(),
                            sizes.size());
  }
  template <class PushType, class T>
  Algorithm makeAlgorithm(std::string spirvname,
                          std::vector<MetaBuffer*> buffers,
                          const T specConsts) {
    constexpr size_t nSpecConsts = boost::pfr::tuple_size_v<T>;
    std::array<u32, nSpecConsts> sizes;
    constexpr_for<0, nSpecConsts, 1>([&sizes](auto i) {
      sizes[i] = sizeof(boost::pfr::tuple_element_t<i, T>);
    });
    constexpr size_t nPushConsts = boost::pfr::tuple_size_v<PushType>;
    std::array<u32, nPushConsts> pushSizes;
    constexpr_for<0, nPushConsts, 1>([&pushSizes](auto i) {
      pushSizes[i] = sizeof(boost::pfr::tuple_element_t<i, PushType>);
    });
    return makeAlgorithmRaw(spirvname, buffers,
                            bit_cast<const u8*>(&specConsts), sizes.data(),
                            sizes.size(), pushSizes.data(), pushSizes.size());
  }
  RaiiVkFFTConf makeFFTConf(const MetaBuffer& buffer, std::array<u32, 3> dims,
                            u32 numberBatches = 1) {
    RaiiVkFFTConf ret{};
    ret.conf.device = ((VkDevice*)&device);
    ret.conf.FFTdim = 1 + dims[1] > 1 ? 1 : 0 + dims[2] > 1 ? 1 : 0;
    ret.conf.size[0] = dims[0];
    ret.conf.size[1] = dims[1];
    ret.conf.size[2] = dims[2];
    ret.conf.numberBatches = numberBatches;
    ret.conf.physicalDevice = (VkPhysicalDevice*)&physicalDevice;
    ret.conf.queue = (VkQueue*)&queue;
    ret.conf.commandPool = (VkCommandPool*)&commandPool;
    ret.conf.fence = (VkFence*)&fence;
    ret.conf.buffer = (VkBuffer*)&buffer.buffer;
    ret.bufferSizes = {buffer.aInfo.size};
    ret.conf.bufferSize = ret.bufferSizes.data();
    return ret;
  }
  ~Manager();
};

struct Renderer {
  // non-owned
  Manager* p_mgr;
  //  owned
  vk::RenderPass renderPass;
  vk::Pipeline graphicsPipeline;
  vk::PipelineLayout graphicsPipelineLayout;
  vk::SwapchainKHR swapchain;
  std::vector<vk::Framebuffer> swapChainFrameBuffers;
  std::vector<vk::Image> swapChainImages;
  vk::Format swapChainImageFormat;
  vk::Extent2D swapChainExtent;
  std::array<u32, 2> render_queue_indices = {UINT32_MAX, UINT32_MAX};
  vk::Queue graphicsQueue;
  vk::Queue presentQueue;
  std::vector<vk::ImageView> swapChainImageViews;
  std::vector<vk::Semaphore> imageAvailableSemaphores;
  std::vector<vk::Semaphore> renderFinishedSemaphores;
  std::vector<vk::Fence> imageInFlightFences;
  std::vector<vk::Fence> inFlightFences;
  vk::CommandPool command_pool;
  std::vector<vk::CommandBuffer> commandBuffers;
  vk::CommandBuffer reduction_buffer;
  MetaBuffer vertexBuffer;
  AllocatedImage colormap_img;
  MetaBuffer colormap;
  vk::ImageView colormap_view;
  vk::Sampler colormap_sampler;
  vk::DescriptorSetLayout descriptorSetLayout;
  vk::DescriptorPool descriptorPool;
  vk::DescriptorSet descriptorSet;
  vk::SurfaceCapabilitiesKHR capabilities;
  vk::SurfaceFormatKHR surface_format;
  vk::PresentModeKHR present_mode;
  MetaBuffer value_buffer;
  MetaBuffer minmax_buffer;
  Algorithm first_minmax_reduction;
  Algorithm minmax_reduction;
  Algorithm fill_colormap_img;
  u32 n_images;
  bool frameBufferResized;
  u32 currentFrame = 0;
  Renderer(Manager& manager, u32 nx, u32 ny);
  void cleanupSwapchain();
  void recreateSwapchain();
  void drawFrame();
  ~Renderer();
};

template <class T>
void writeCsv(std::ofstream& of, T* v, u32 nColumns, u32 nRows = 1,
              const std::vector<std::string>& heading = {}) {
  std::string out;
  if (heading.size()) {
    for (const auto& h : heading) {
      of << h << ' ';
    }
    of << '\n';
  }
  for (u32 j = 0; j < nRows; j++) {
    for (u32 i = 0; i < nColumns; i++) {
      of << numfmt(v[j * nColumns + i]) << ' ';
    }
    of << '\n';
  }
  of.close();
}

std::vector<u32> readFile(const std::string& filename);
vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance,
                                      const s32 desiredGPU = -1);

template <typename Func>
void oneTimeSubmit(const vk::Device& device, const vk::CommandPool& commandPool,
                   const vk::Queue& queue, const Func& func) {
  vk::CommandBuffer commandBuffer =
      device
          .allocateCommandBuffers(
              {commandPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
  commandBuffer.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  func(commandBuffer);
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  queue.submit(submitInfo, nullptr);
  queue.waitIdle();
}

void appendOp(vk::CommandBuffer& b, Algorithm& a, u32 X, u32 Y, u32 Z);
void appendOpNoBarrier(vk::CommandBuffer& b, Algorithm& a, u32 X, u32 Y = 1,
                       u32 Z = 1);
