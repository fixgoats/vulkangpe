#include "hack.h"
#include "vk_mem_alloc.h"
#include <cmath>
#include <cstring>
#include <cxxopts.hpp>
#include <iostream>

#include "SDL3/SDL.h"
#include "SDL3/SDL_vulkan.h"
#include "colormaps.hpp"
#include "mathhelpers.h"
#include "typedefs.h"
#include "vkcore.h"
#include <random>
#include <toml++/toml.hpp>
#include <vulkan/vulkan.h>

using std::bit_cast;

static const char* BasePath = SDL_GetBasePath();

SDL_Window* create_window_sdl(const char* window_name = "", u32 flags = 0) {
  SDL_Init(SDL_INIT_VIDEO);
  if (!SDL_Vulkan_LoadLibrary(nullptr)) {
    SDL_Log("Unable to load Vulkan library: %s", SDL_GetError());
  };
  SDL_Window* window =
      SDL_CreateWindow(window_name, 645, 480, SDL_WINDOW_VULKAN | flags);
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

constexpr u32 nX = 512;
constexpr u32 nY = 512;
constexpr u32 gX = 8;
constexpr u32 gY = 8;
constexpr struct SimConstants {
  u32 nx = nX;
  u32 ny = nY;
  u32 gx = gX;
  u32 gy = gY;
  f32 alpha = 0.0004;
  f32 gammalp = 0.2;
  f32 Gamma = 0.1;
  f32 G = 0.002;
  f32 R = 0.016;
  f32 eta = 2;
  f32 dt = 0.1;

  bool validate() { return !((nx % gx) | (ny % gy)); }
  constexpr u32 X() const { return nx / gx; }

  constexpr u32 Y() const { return ny / gy; }
} sc;

f32 pumpProfile(f32 x, f32 y, f32 L, f32 r, f32 beta) {
  return square(square(L)) /
         (square(square(x) + square(beta * y) - square(r)) + square(square(L)));
}

f32 int_to_coord(u32 i, u32 nx, f32 start, f32 end) {
  return (end - start) * static_cast<f32>(i) / static_cast<f32>(nx) + start;
}

int main(int argc, char* argv[]) {
  std::cout << "started\n";
  Manager mgr(10 * 1024 * 1024);
  std::cout << "created manager\n";
  std::vector<f32> cpu_vec1(100, 1);
  std::vector<f32> cpu_vec2(100, 3);
  MetaBuffer buf1 = mgr.vecToBuffer(cpu_vec1);
  std::cout << "made 1st buffer\n";
  MetaBuffer buf2 = mgr.vecToBuffer(cpu_vec2);
  std::cout << "made 2nd buffer\n";
  MetaBuffer resultbuf = mgr.makeRawBuffer<f32>(100);
  std::cout << "made result buffer\n";
  Algorithm add = mgr.makeAlgorithmRaw("Shaders/hello-world.spv", {},
                                       {&buf1, &buf2, &resultbuf});
  std::cout << "made algorithm\n";
  vk::CommandBuffer cb = mgr.beginRecord();
  appendOp(cb, add, 100, 1, 1);
  cb.end();
  std::cout << "recorded command buffer\n";
  mgr.execute(cb);
  std::cout << "executed command buffer\n";
  std::vector<f32> resultvec(100);
  mgr.writeFromBuffer(resultbuf, resultvec);
  for (const auto& e : resultvec) {
    std::cout << e << ' ';
  }
  std::cout << std::endl;
  /*auto window = create_window_sdl("Bleh", SDL_WINDOW_RESIZABLE);
  {
    Manager mgr(10 * 1024 * 1024, window);
    Renderer renderer(mgr, sc.nx, sc.ny);
    std::vector<c32> cpu_psir(sc.nx * sc.ny);
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<f32> dis(-0.001, 0.001);
    for (auto& x : cpu_psir) {
      x = c32{dis(gen), dis(gen)};
    }
    constexpr f32 xstart = -48.0;
    constexpr f32 xend = 48.0;
    constexpr f32 dx = (xend - xstart) / sc.nx;
    constexpr f32 kmax = M_PI / dx;
    std::vector<c32> cpu_kProp(sc.nx * sc.ny);
    for (u32 j = 0; j < sc.ny; j++) {
      f32 ky = int_to_coord(j, sc.ny, -kmax, kmax);
      for (u32 i = 0; i < sc.nx; i++) {
        f32 kx = int_to_coord(i, sc.nx, -kmax, kmax);
        cpu_kProp[j * sc.nx + i] = std::exp(
            c32{0.0, -0.5f * hbar * sc.dt * (square(kx) + square(ky)) / 0.32f});
      }
    }
    MetaBuffer psir = mgr.vecToBuffer(cpu_psir);
    MetaBuffer oldPsir = mgr.makeRawBuffer<c32>(sc.nx * sc.ny);
    MetaBuffer psik = mgr.makeRawBuffer<c32>(sc.nx * sc.ny);
    MetaBuffer nR = mgr.makeRawBuffer<f32>(sc.nx * sc.ny);
    mgr.defaultInitBuffer<f32>(nR, sc.nx * sc.ny);
    MetaBuffer kTimeEvo = mgr.vecToBuffer(cpu_kProp);
    std::vector<f32> cpu_pump(sc.nx * sc.ny);
    for (u32 j = 0; j < sc.ny; j++) {
      f32 y = int_to_coord(j, sc.ny, xstart, xend);
      for (u32 i = 0; i < sc.nx; i++) {
        f32 x = int_to_coord(i, sc.nx, xstart, xend);
        std::cout << x << ' ';
        cpu_pump[j * sc.nx + i] = 10.0 * pumpProfile(x, y, 1.8, 4.3, 0.9);
      }
    }
    MetaBuffer pump = mgr.vecToBuffer(cpu_pump);

    Algorithm rstep = mgr.makeAlgorithm("Shaders/rstep.spv", {},
                                        {&psir, &oldPsir, &nR, &pump}, sc);
    Algorithm kstep =
        mgr.makeAlgorithm("Shaders/kstep.spv", {}, {&psik, &kTimeEvo}, sc);
    Algorithm finalstep = mgr.makeAlgorithm("Shaders/finalstep.spv", {},
                                            {&psir, &oldPsir, &nR, &pump}, sc);
    Algorithm transfer = mgr.makeAlgorithm("Shaders/transferandsquare.spv", {},
                                           {&psir, &renderer.value_buffer}, sc);
    VkFFTConfiguration conf{};
    conf.device = pcast<VkDevice>(&mgr.device);
    conf.FFTdim = 2;
    conf.size[0] = sc.nx;
    conf.size[1] = sc.ny;
    conf.queue = pcast<VkQueue>(&mgr.queue);
    conf.fence = pcast<VkFence>(&mgr.fence);
    conf.commandPool = pcast<VkCommandPool>(&mgr.commandPool);
    conf.physicalDevice = pcast<VkPhysicalDevice>(&mgr.physicalDevice);
    conf.buffer = pcast<VkBuffer>(&psik.buffer);
    conf.isInputFormatted = true;
    conf.inputBuffer = pcast<VkBuffer>(&psir.buffer);
    conf.bufferSize = &psik.aInfo.size;
    conf.inputBufferSize = &psir.aInfo.size;
    conf.inverseReturnToInputBuffer = true;
    conf.normalize = true;
    VkFFTApplication app{};
    auto resFFT = initializeVkFFT(&app, conf);
    if (resFFT != VKFFT_SUCCESS) {
      std::cout << resFFT << '\n';
      exit(1);
    }
    auto cb = mgr.beginRecord();
    VkFFTLaunchParams launchParams{};
    launchParams.commandBuffer = pcast<VkCommandBuffer>(&cb);
    for (int i = 0; i < 100; i++) {
      appendOp(cb, rstep, sc.X(), sc.Y(), 1);
      resFFT = VkFFTAppend(&app, -1, &launchParams);
      if (resFFT != VKFFT_SUCCESS) {
        std::cout << resFFT << '\n';
        exit(1);
      }
      appendOp(cb, kstep, sc.X(), sc.Y(), 1);
      resFFT = VkFFTAppend(&app, 1, &launchParams);
      if (resFFT != VKFFT_SUCCESS) {
        std::cout << resFFT << '\n';
        exit(1);
      }
      appendOp(cb, finalstep, sc.X(), sc.Y(), 1);
    }
    cb.end();
    auto xferCommand = mgr.beginRecord();
    appendOp(xferCommand, transfer, sc.X(), sc.Y(), 1);
    xferCommand.end();

    mgr.execute(cb);
    mgr.writeFromBuffer(psir, cpu_psir);
    for (const auto& x : cpu_psir) {
      std::cout << x << ' ';
    }
    bool running = true;
    while (running) {
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT ||
            (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
             event.window.windowID == SDL_GetWindowID(window))) {
          running = false;
          break;
        }
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
          SDL_Delay(10);
          continue;
        }
      }
      mgr.execute(cb);
      mgr.execute(xferCommand);
      renderer.drawFrame();
    }
    mgr.writeFromBuffer(psir, cpu_psir);
    for (const auto& x : cpu_psir) {
      std::cout << x << ' ';
    }

    deleteVkFFT(&app);
  }

  SDL_DestroyWindow(window);*/
  return 0;
}
