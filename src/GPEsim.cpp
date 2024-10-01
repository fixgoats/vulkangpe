#include "GPEsim.hpp"
VulkanApp::VulkanApp(SimConstants consts) {
  nSpecConsts = 11;
  nComplexBuffers = 3;
  nFloatBuffers = 2;
  params = consts;
  vk::ApplicationInfo appInfo{appName.c_str(), 1, nullptr, 0,
                              VK_API_VERSION_1_3};
#ifndef DEBUG
  const std::vector<const char*> layers;
#else
  const std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
#endif // DEBUG
  vk::InstanceCreateInfo iCI(vk::InstanceCreateFlags(), &appInfo, layers, {});
  instance = vk::createInstance(iCI);
  pDevice = pickPhysicalDevice(instance);
  uint32_t computeQueueFamilyIndex = getComputeQueueFamilyIndex();
  float queuePriority = 1.0f;
  vk::DeviceQueueCreateInfo dQCI(vk::DeviceQueueCreateFlags(),
                                 computeQueueFamilyIndex, 1, &queuePriority);
  vk::DeviceCreateInfo dCI(vk::DeviceCreateFlags(), dQCI);
  device = pDevice.createDevice(dCI);
  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  computeQueueFamilyIndex);
  commandPool = device.createCommandPool(commandPoolCreateInfo);
  commandBuffer = device
                      .allocateCommandBuffers(
                          {commandPool, vk::CommandBufferLevel::ePrimary, 1})
                      .front();
  queue = device.getQueue(computeQueueFamilyIndex, 0);
  fence = device.createFence(vk::FenceCreateInfo());
  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.physicalDevice = pDevice;
  allocatorInfo.vulkanApiVersion = pDevice.getProperties().apiVersion;
  allocatorInfo.device = device;
  allocatorInfo.instance = instance;
  vmaCreateAllocator(&allocatorInfo, &allocator);
  uint32_t nElements = nElementsX * nElementsY;
  vk::BufferCreateInfo stagingBCI({}, nElements * sizeof(c32),
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst);
  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  staging.allocate(allocator, allocCreateInfo, stagingBCI);
  vk::BufferCreateInfo stateBCI{vk::BufferCreateFlags(),
                                nElements * sizeof(c32),
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst |
                                    vk::BufferUsageFlagBits::eTransferSrc,
                                vk::SharingMode::eExclusive,
                                1,
                                &computeQueueFamilyIndex};
  vk::BufferCreateInfo floatBCI{vk::BufferCreateFlags(),
                                nElements * sizeof(float),
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst |
                                    vk::BufferUsageFlagBits::eTransferSrc,
                                vk::SharingMode::eExclusive,
                                1,
                                &computeQueueFamilyIndex};
  allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocCreateInfo.priority = 1.0f;
  for (uint32_t i = 0; i < nComplexBuffers; i++) {
    computeBuffers.emplace_back(allocator, allocCreateInfo, stateBCI);
  }
  for (uint32_t i = 0; i < nFloatBuffers; i++) {
    computeBuffers.emplace_back(allocator, allocCreateInfo, floatBCI);
  }
  std::vector<std::string> moduleNames = {"rstep.spv", "kstep.spv",
                                          "finalstep.spv"};
  conf.device = (VkDevice*)&device;
  conf.FFTdim = 2;
  conf.size[0] = params.nElementsX;
  conf.size[1] = params.nElementsY;
  conf.queue = reinterpret_cast<VkQueue*>(&queue);
  conf.fence = reinterpret_cast<VkFence*>(&fence);
  conf.commandPool = reinterpret_cast<VkCommandPool*>(&commandPool);
  conf.physicalDevice = reinterpret_cast<VkPhysicalDevice*>(&pDevice);
  conf.buffer = reinterpret_cast<VkBuffer*>(&computeBuffers[0].buffer);
  conf.bufferSize = &computeBuffers[0].aInfo.size;
  conf.normalize = 1;
  auto resFFT = initializeVkFFT(&app, conf);
  setupPipelines(moduleNames);
}

void VulkanApp::copyBuffers(vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
                            uint32_t bufferSize) {
  commandBuffer.reset();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);

  commandBuffer.begin(cBBI);
  commandBuffer.copyBuffer(srcBuffer, dstBuffer,
                           vk::BufferCopy(0, 0, bufferSize));
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, commandBuffer);
  queue.submit(submitInfo, fence);
  queue.waitIdle();
  auto result = device.waitForFences({fence}, true, -1);

  device.freeCommandBuffers(commandPool, 1, &commandBuffer);
}

void VulkanApp::runSim(uint32_t n) {
  commandBuffer.reset();
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  VkFFTLaunchParams launchParams{};
  launchParams.commandBuffer =
      reinterpret_cast<VkCommandBuffer*>(&commandBuffer);

  for (uint32_t i = 0; i < n; i++) {
    appendPipeline(0);
    auto resFFT = VkFFTAppend(&app, -1, &launchParams);
    appendPipeline(1);
    resFFT = VkFFTAppend(&app, 1, &launchParams);
    appendPipeline(2);
  }
  commandBuffer.end();

  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
  queue.submit(submitInfo, fence);
  queue.waitIdle();
  auto result = device.waitForFences({fence}, true, -1);
  device.freeCommandBuffers(commandPool, 1, &commandBuffer);
}

void VulkanApp::initPsiR() {
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-0.1, 0.1);
  c32* sStagingPtr = reinterpret_cast<c32*>(staging.aInfo.pMappedData);
  for (uint32_t j = 0; j < nElementsY; j++) {
    for (uint32_t i = 0; i < nElementsX; i++) {
      sStagingPtr[j * params.nElementsX + i] = c32{dis(gen), dis(gen)};
    }
  }

  copyBuffers(staging.buffer, computeBuffers[0].buffer,
              params.elementsTotal() * sizeof(c32));
}

void VulkanApp::appendPipeline(uint32_t i) {
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                fullMemoryBarrier, nullptr, nullptr);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                             computePipelines[i]);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   pipelineLayout, 0, {descriptorSets[0]}, {});
  commandBuffer.dispatch(params.X(), params.Y(), 1);
  // commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
  //                               vk::PipelineStageFlagBits::eAllCommands,
  //                               {}, fullMemoryBarrier, nullptr, nullptr);
}

void VulkanApp::initPump() {
  float l = 2.2;
  float r = 5.1;
  float beta = 1.0;
  float* fStagingPtr = reinterpret_cast<float*>(staging.aInfo.pMappedData);
  for (uint32_t j = 0; j < nElementsY; j++) {
    float y = (float)j * dY + startY;
    for (uint32_t i = 0; i < nElementsX; i++) {
      float x = (float)i * dX + startX;
      fStagingPtr[j * nElementsX + i] =
          p * (pumpProfile(x - 12.2, y, l, r, beta) +
               pumpProfile(x + 12.2, y, l, r, beta));
    }
  }
  copyBuffers(staging.buffer, computeBuffers[4].buffer,
              params.elementsTotal() * sizeof(float));
}

void VulkanApp::initKTimeEvo() {
  c32* sStagingPtr = reinterpret_cast<c32*>(staging.aInfo.pMappedData);
  for (uint32_t j = 0; j < nElementsY; j++) {
    float kY = (float)fftshiftidx(j, nElementsY) * dKy + startKy;
    for (uint32_t i = 0; i < nElementsX; i++) {
      float kX = (float)fftshiftidx(i, nElementsX) * dKx + startKx;
      sStagingPtr[j * params.nElementsX + i] =
          std::exp(c32{0., -(0.5f * hbar * dt / m) * (kY * kY + kX * kX)});
    }
  }
  copyBuffers(staging.buffer, computeBuffers[1].buffer,
              params.elementsTotal() * sizeof(c32));
}

void VulkanApp::initNR() {
  float* fStagingPtr = reinterpret_cast<float*>(staging.aInfo.pMappedData);
  for (uint32_t i = 0; i < params.elementsTotal(); i++) {
    fStagingPtr[i] = 0.;
  }
  copyBuffers(staging.buffer, computeBuffers[3].buffer,
              params.elementsTotal() * sizeof(float));
}

void VulkanApp::initBuffers() {
  initPsiR();
  initPump();
  initKTimeEvo();
  initNR();
}

uint32_t VulkanApp::getComputeQueueFamilyIndex() {
  auto queueFamilyProps = pDevice.getQueueFamilyProperties();
  auto propIt =
      std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(),
                   [](const vk::QueueFamilyProperties& prop) {
                     return prop.queueFlags & vk::QueueFlagBits::eCompute;
                   });
  return std::distance(queueFamilyProps.begin(), propIt);
}

void VulkanApp::setupPipelines(std::vector<std::string> moduleNames) {
  for (const auto& name : moduleNames) {
    auto shaderCode = readFile(name);
    vk::ShaderModuleCreateInfo shaderMCI(vk::ShaderModuleCreateFlags(),
                                         shaderCode);
    modules.emplace_back(device.createShaderModule(shaderMCI));
  }
  std::vector<vk::DescriptorSetLayoutBinding> dSLBs;
  for (uint32_t i = 0; i < computeBuffers.size(); i++) {
    dSLBs.emplace_back(i, vk::DescriptorType::eStorageBuffer, 1,
                       vk::ShaderStageFlagBits::eCompute);
  }
  vk::DescriptorSetLayoutCreateInfo dSLCI(vk::DescriptorSetLayoutCreateFlags(),
                                          dSLBs);
  dSL = device.createDescriptorSetLayout(dSLCI);
  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(), dSL);
  pipelineLayout = device.createPipelineLayout(pLCI);
  pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());
  std::vector<vk::SpecializationMapEntry> bleh(nSpecConsts);
  for (uint32_t i = 0; i < nSpecConsts; i++) {
    bleh[i].constantID = i;
    bleh[i].offset = i * 4;
    bleh[i].size = 4;
  }
  vk::SpecializationInfo specInfo;
  specInfo.mapEntryCount = nSpecConsts;
  specInfo.pMapEntries = bleh.data();
  specInfo.dataSize = sizeof(SimConstants);
  specInfo.pData = &params;

  for (const auto& mod : modules) {
    vk::PipelineShaderStageCreateInfo cSCI(vk::PipelineShaderStageCreateFlags(),
                                           vk::ShaderStageFlagBits::eCompute,
                                           mod, "main", &specInfo);
    vk::ComputePipelineCreateInfo cPCI(vk::PipelineCreateFlags(), cSCI,
                                       pipelineLayout);
    auto result = device.createComputePipeline(pipelineCache, cPCI);
    assert(result.result == vk::Result::eSuccess);
    computePipelines.push_back(result.value);
  }

  vk::DescriptorPoolSize dPS(vk::DescriptorType::eStorageBuffer, 1);
  vk::DescriptorPoolCreateInfo dPCI(
      vk::DescriptorPoolCreateFlags(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
      1, dPS);
  descriptorPool = device.createDescriptorPool(dPCI);
  vk::DescriptorSetAllocateInfo dSAI(descriptorPool, 1, &dSL);
  descriptorSets = device.allocateDescriptorSets(dSAI);
  descriptorSet = descriptorSets[0];
  std::vector<vk::DescriptorBufferInfo> dBIs;
  for (const auto& b : computeBuffers) {
    dBIs.emplace_back(b.buffer, 0, b.aInfo.size);
  }
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  for (uint32_t i = 0; i < dBIs.size(); i++) {
    writeDescriptorSets.emplace_back(descriptorSet, i, 0, 1,
                                     vk::DescriptorType::eStorageBuffer,
                                     nullptr, &dBIs[i]);
  }
  device.updateDescriptorSets(writeDescriptorSets, {});
}

VulkanApp::~VulkanApp() {
  device.waitIdle();
  deleteVkFFT(&app);
  device.destroyFence(fence);
  for (auto& p : computePipelines) {
    device.destroyPipeline(p);
  }
  device.destroyPipelineCache(pipelineCache);
  device.destroyDescriptorPool(descriptorPool);
  for (auto& m : modules) {
    device.destroyShaderModule(m);
  }
  device.destroyPipelineLayout(pipelineLayout);
  device.destroyDescriptorSetLayout(dSL);
  staging.extirpate(allocator);
  for (auto& b : computeBuffers) {
    b.extirpate(allocator);
  }
  vmaDestroyAllocator(allocator);
  device.destroyCommandPool(commandPool);
  device.destroy();
  instance.destroy();
}
