#pragma once
#include "hack.hpp"
#include "mathhelpers.hpp"
#include "vkFFT.h"
#include "vkhelpers.hpp"
#include <random>

static const std::string appName{"Vulkan GPE Simulator"};

constexpr float hbar = 6.582119569e-1;
constexpr uint32_t nElementsX = 512;
constexpr uint32_t nElementsY = 512;
constexpr uint32_t xGroupSize = 16;
constexpr uint32_t yGroupSize = 16;
constexpr float startX = -64.;
constexpr float endX = 64.;
constexpr float startY = -64.;
constexpr float endY = 64.;
constexpr float dX = (endX - startX) / (float)nElementsX;
constexpr float dY = (endY - startY) / (float)nElementsY;
constexpr float endKx = M_PI / dX;
constexpr float startKx = -endKx;
constexpr float endKy = M_PI / dY;
constexpr float startKy = -endKy;
constexpr float dKx = 2 * M_PI / (endX - startX);
constexpr float dKy = 2 * M_PI / (endY - startY);

constexpr float alpha = 0.0004;
constexpr float gammalp = 0.2;
constexpr float Gamma = 0.1;
constexpr float G = 0.002;
constexpr float R = 0.016;
constexpr float eta = 2.;
constexpr float dt = 0.05;
constexpr float m = 0.32;
constexpr float p = 9.2;

struct VulkanApp {
  vk::Instance instance;
  vk::PhysicalDevice pDevice;
  vk::Device device;
  vk::Queue queue;
  vk::Fence fence;
  VmaAllocator allocator;
  MetaBuffer staging;
  uint32_t nComplexBuffers;
  uint32_t nFloatBuffers;
  uint32_t nSpecConsts; // same as # of field on params, don't have the
                        // willpower to introduce metaprogramming reflections
  SimConstants params;
  std::vector<MetaBuffer> computeBuffers;
  std::vector<vk::ShaderModule> modules;
  vk::DescriptorSetLayout dSL;
  vk::PipelineLayout pipelineLayout;
  vk::PipelineCache pipelineCache;
  std::vector<vk::Pipeline> computePipelines;
  std::vector<vk::DescriptorSet> descriptorSets;
  vk::DescriptorSet descriptorSet;
  vk::DescriptorPool descriptorPool;
  vk::CommandPool commandPool;
  vk::CommandBuffer commandBuffer;
  VkFFTConfiguration conf{};
  VkFFTApplication app{};

  std::random_device rd;

  VulkanApp(SimConstants consts);
  ~VulkanApp();
  void copyBuffers(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                   uint32_t bufferSize);
  void runSim(uint32_t n);
  void initPsiR();
  void appendPipeline(uint32_t i);
  void initPump();
  void initKTimeEvo();
  void initNR();
  void initBuffers();
  uint32_t getComputeQueueFamilyIndex();
  void setupPipelines(std::vector<std::string> moduleNames);
  template <typename T>
  std::vector<T> outputBuffer(uint32_t n) {
    T* sStagingPtr = reinterpret_cast<T*>(staging.aInfo.pMappedData);
    copyBuffers(computeBuffers[n].buffer, staging.buffer,
                computeBuffers[n].aInfo.size);
    std::vector<T> retVec(params.elementsTotal());
    memcpy(retVec.data(), sStagingPtr, computeBuffers[n].aInfo.size);
    return std::move(retVec);
  }
};
