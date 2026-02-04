#include "hack.hpp"
#include "vkFFT.h"
#include <cmath>
#include <cstring>
#include <cxxopts.hpp>
#include <iostream>
#include <toml++/toml.hpp>

// #include "colormaps.hpp"
#include "Eigen/Dense"
#include "logger.hpp"
#include "mathhelpers.hpp"
#include "typedefs.hpp"
#include "vkcore.hpp"
#include <H5Include.h>
#include <fstream>
#include <iterator>
#include <random>
#include <ranges>
#include <toml++/toml.hpp>
#include <vulkan/vulkan.h>
using Eigen::Vector2d;

// using std::bit_cast;
// Number of threads in subgroups is 32 on nvidia and some newer amd systems
// 64 in older amd systems but also some newer ones.
// constexpr u32 WAVE_SIZE = 32;

struct Dispatch {
  u32 nx;
  u32 ny;
  u32 nz;
  u32 xgroups;
  u32 ygroups;
  u32 zgroups;

  u32 X() { return (nx + xgroups - 1) / xgroups; }
  u32 Y() { return (ny + ygroups - 1) / ygroups; }
  u32 Z() { return (nz + zgroups - 1) / zgroups; }
};

// SDL_Window* create_window_sdl(const char* window_name = "", u32 flags = 0) {
//   SDL_Init(SDL_INIT_VIDEO);
//   if (!SDL_Vulkan_LoadLibrary(nullptr)) {
//     SDL_Log("Unable to load Vulkan library: %s", SDL_GetError());
//   };
//   SDL_Window* window =
//       SDL_CreateWindow(window_name, 645, 480, SDL_WINDOW_VULKAN | flags);
//   if (!window) {
//     SDL_Log("CreateWindow failed with error: %s", SDL_GetError());
//   }
//   return window;
// }
//
// vk::SurfaceKHR create_surface_sdl(VkInstance instance, SDL_Window* window,
//                                   VkAllocationCallbacks* allocator = nullptr)
//                                   {
//   vk::SurfaceKHR surf;
//   if (!SDL_Vulkan_CreateSurface(window, instance, allocator,
//                                 bit_cast<VkSurfaceKHR*>(&surf))) {
//     SDL_Log("CreateSurface failed with error: %s", SDL_GetError());
//   }
//   return surf;
// }

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
  f32 dt = 0.02;

  bool validate() { return !((nx % gx) | (ny % gy)); }
  constexpr u32 X() const { return nx / gx; }
  constexpr u32 Y() const { return ny / gy; }
} sc;

// f32 pumpProfile(f32 x, f32 y, f32 L, f32 r, f32 beta) {
//   return square(square(L)) /
//          (square(square(x) + square(beta * y) - square(r)) +
//          square(square(L)));
// }

f32 pumpProfile(f32 x, f32 y, f32 sigma) {
  return std::exp(-(x * x + y * y) / (sigma * sigma));
}

f32 int_to_coord(u32 i, u32 nx, f32 start, f32 end) {
  return (end - start) * static_cast<f32>(i) / static_cast<f32>(nx) + start;
}
struct Line {
  friend std::istream& operator>>(std::istream& is, Line& line) {
    return std::getline(is, line.lineTemp);
  }

  // Output function.
  friend std::ostream& operator<<(std::ostream& os, const Line& line) {
    return os << line.lineTemp;
  }

  // cast to needed result
  operator std::string() const { return lineTemp; }
  // Temporary Local storage for line
  std::string lineTemp{};
};

std::vector<Vector2d> readPoints(const std::string& fname) {
  u32 m = 0;
  std::ifstream f(fname);
  if (!f.good()) {
    runtime_exc("File {} doesn't exist or couldn't be opened", fname);
  }

  std::vector<std::string> allLines{std::istream_iterator<Line>(f),
                                    std::istream_iterator<Line>()};
  m = allLines.size();

  std::vector<Vector2d> M(m);
  for (u32 j = 0; j < m; j++) {
    std::istringstream stream(allLines[j]);
    std::vector<f64> v{std::istream_iterator<f64>(stream),
                       std::istream_iterator<f64>()};
    M[j] = {v[0], v[1]};
  }
  return M;
}

template <class T>
struct RangeConf {
  T start;
  T end;
  u64 n;

  constexpr T d() const { return (end - start) / n; }
  constexpr T ith(uint i) const { return start + i * d(); }
};

RangeConf<Vector2d> tblToVecRange(const toml::table& tbl) {
  toml::array start = *tbl["start"].as_array();
  toml::array end = *tbl["end"].as_array();
  return {{start[0].value<f64>().value(), start[1].value<f64>().value()},
          {end[0].value<f64>().value(), end[1].value<f64>().value()},
          tbl["n"].value_or<u64>(0)};
}

template <class T>
RangeConf<T> tblToRange(toml::table& tbl) {
  return {tbl["start"].value_or<T>(0.0), tbl["end"].value_or<T>(0.0),
          tbl["n"].value_or<u64>(0)};
}

struct SimConf {
  std::string pointPath;
  std::string output;
  RangeConf<f32> xrange;
  RangeConf<f32> yrange;
  u32 substeps;
  u32 steps;
  f32 m;
  SimConstants sc;
};

#define SET_STRUCT_FIELD(c, tbl, key)                                          \
  if (tbl.contains(#key))                                                      \
  c.key = *tbl[#key].value<decltype(c.key)>()

std::optional<SimConf> tomlToSimConf(const std::string& tomlPath) {
  logDebug("Function: tomlToSimConf.");
  toml::table tbl;
  try {
    tbl = toml::parse_file(tomlPath);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << tomlPath
              << " failed with exception: " << err.what() << '\n';
    return {};
  }
  SimConf conf{};

  SET_STRUCT_FIELD(conf, tbl, substeps);
  logDebug(std::format("substeps set to {}", conf.substeps));
  SET_STRUCT_FIELD(conf, tbl, steps);
  logDebug(std::format("steps set to {}", conf.steps));
  SET_STRUCT_FIELD(conf, tbl, pointPath);
  logDebug(std::format("pointPath set to {}", conf.pointPath));
  SET_STRUCT_FIELD(conf, tbl, output);
  logDebug(std::format("output set to {}", conf.output));
  SET_STRUCT_FIELD(conf, tbl, m);
  logDebug(std::format("m set to {}", conf.m));
  SET_STRUCT_FIELD(conf.sc, tbl, dt);
  logDebug(std::format("sc.dt set to {}", conf.sc.dt));
  SET_STRUCT_FIELD(conf.sc, tbl, alpha);
  logDebug(std::format("sc.alpha set to {}", conf.sc.alpha));
  SET_STRUCT_FIELD(conf.sc, tbl, gammalp);
  logDebug(std::format("sc.gammalp set to {}", conf.sc.gammalp));
  SET_STRUCT_FIELD(conf.sc, tbl, Gamma);
  logDebug(std::format("sc.Gamma set to {}", conf.sc.Gamma));
  SET_STRUCT_FIELD(conf.sc, tbl, G);
  logDebug(std::format("sc.G set to {}", conf.sc.G));
  SET_STRUCT_FIELD(conf.sc, tbl, R);
  logDebug(std::format("sc.R set to {}", conf.sc.R));
  SET_STRUCT_FIELD(conf.sc, tbl, eta);
  logDebug(std::format("sc.eta set to {}", conf.sc.eta));
  SET_STRUCT_FIELD(conf.sc, tbl, gx);
  logDebug(std::format("sc.gx set to {}", conf.sc.gx));
  SET_STRUCT_FIELD(conf.sc, tbl, gy);
  logDebug(std::format("sc.gy set to {}", conf.sc.gy));
  conf.xrange = tblToRange<f32>(*tbl["xrange"].as_table());
  conf.yrange = tblToRange<f32>(*tbl["yrange"].as_table());
  conf.sc.nx = conf.xrange.n;
  logDebug(std::format("Set sc.nx to {}", conf.sc.nx));
  conf.sc.ny = conf.yrange.n;
  logDebug(std::format("Set sc.ny to {}", conf.sc.ny));

  logDebug("Exiting tomlToSimConf.");
  return conf;
}
#undef SET_STRUCT_FIELD

int main(int argc, char* argv[]) {
  // int ret_val = test_graphical();
  cxxopts::Options options("MyProgram", "bleh");
  options.add_options()("c,conf", "Configuration file",
                        cxxopts::value<std::string>());

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }

  if (result["c"].count()) {
    std::string fname = result["c"].as<std::string>();
    SimConf conf;
    if (auto opt = tomlToSimConf(fname); opt.has_value()) {
      conf = opt.value();
    } else {
      return EXIT_FAILURE;
    }
    u32 totalCells = conf.sc.nx * conf.sc.ny;

    logDebug(std::format("totalCells: {}", totalCells));
    Manager mgr(10 * 1024 * 1024);
    std::vector<c32> cpu_psir(totalCells);
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<f32> dis(-1e-8, 1e-8);
    for (auto& x : cpu_psir) {
      x = c32{dis(gen), dis(gen)};
    }
    const f32 kmax = M_PI / conf.xrange.d();
    const RangeConf<f32> kxrange{-kmax, kmax, conf.sc.nx};
    const RangeConf<f32> kyrange{-kmax, kmax, conf.sc.ny};
    std::vector<c32> cpu_kProp(totalCells);
    for (u32 j = 0; j < conf.sc.ny; j++) {
      f32 ky = kyrange.ith(j);
      for (u32 i = 0; i < conf.sc.nx; i++) {
        f32 kx = kxrange.ith(i);
        cpu_kProp[j * conf.sc.nx + i] = std::exp(c32{
            0.0, -0.5f * hbar * sc.dt * (square(kx) + square(ky)) / conf.m});
      }
    }
    MetaBuffer psir = mgr.vecToBuffer(cpu_psir);
    MetaBuffer oldPsir = mgr.makeRawBuffer<c32>(totalCells);
    MetaBuffer psik = mgr.makeRawBuffer<c32>(totalCells);
    MetaBuffer nR = mgr.makeRawBuffer<f32>(totalCells);
    mgr.defaultInitBuffer<f32>(nR, totalCells);
    MetaBuffer kTimeEvo = mgr.vecToBuffer(cpu_kProp);
    auto points = readPoints(conf.pointPath);
    std::vector<f32> cpu_pump(totalCells);

    for (const auto& point : points) {
      for (u32 j = 0; j < sc.ny; j++) {
        f32 y = conf.yrange.ith(j) - point.y();
        for (u32 i = 0; i < sc.nx; i++) {
          f32 x = conf.xrange.ith(i) - point.x();
          // std::cout << x << ' ';
          cpu_pump[j * sc.nx + i] += 16 * pumpProfile(x, y, 1.3);
        }
      }
    }
    MetaBuffer pump = mgr.vecToBuffer(cpu_pump);

    Algorithm rstep = mgr.makeAlgorithm("Shaders/rstep.spv", 0, 4, conf.sc);
    rstep.bindData({}, {&psir, &oldPsir, &nR, &pump}, {});
    Algorithm kstep = mgr.makeAlgorithm("Shaders/kstep.spv", 0, 2, conf.sc);
    kstep.bindData({}, {&psik, &kTimeEvo}, {});
    Algorithm finalstep =
        mgr.makeAlgorithm("Shaders/finalstep.spv", 0, 4, conf.sc);
    finalstep.bindData({}, {&psir, &oldPsir, &nR, &pump}, {});
    VkFFTConfiguration vkfftconf{};
    vkfftconf.device = pcast<VkDevice>(&mgr.device);
    vkfftconf.FFTdim = 2;
    vkfftconf.size[0] = sc.nx;
    vkfftconf.size[1] = sc.ny;
    vkfftconf.queue = pcast<VkQueue>(&mgr.queue);
    vkfftconf.fence = pcast<VkFence>(&mgr.fence);
    vkfftconf.commandPool = pcast<VkCommandPool>(&mgr.commandPool);
    vkfftconf.physicalDevice = pcast<VkPhysicalDevice>(&mgr.physicalDevice);
    vkfftconf.buffer = pcast<VkBuffer>(&psik.buffer);
    vkfftconf.isInputFormatted = true;
    vkfftconf.inputBuffer = pcast<VkBuffer>(&psir.buffer);
    vkfftconf.bufferSize = &psik.aInfo.size;
    vkfftconf.inputBufferSize = &psir.aInfo.size;
    vkfftconf.inverseReturnToInputBuffer = true;
    vkfftconf.normalize = true;
    VkFFTApplication app{};
    auto resFFT = initializeVkFFT(&app, vkfftconf);
    if (resFFT != VKFFT_SUCCESS) {
      std::cout << resFFT << '\n';
      exit(1);
    }
    auto cb = mgr.beginRecord();
    VkFFTLaunchParams launchParams{};
    launchParams.commandBuffer = pcast<VkCommandBuffer>(&cb);
    for (u32 i = 0; i < conf.substeps; i++) {
      appendOp(cb, rstep, conf.sc.X(), conf.sc.Y(), 1);
      resFFT = VkFFTAppend(&app, -1, &launchParams);
      if (resFFT != VKFFT_SUCCESS) {
        std::cout << resFFT << '\n';
        exit(1);
      }
      appendOp(cb, kstep, conf.sc.X(), conf.sc.Y(), 1);
      resFFT = VkFFTAppend(&app, 1, &launchParams);
      if (resFFT != VKFFT_SUCCESS) {
        std::cout << resFFT << '\n';
        exit(1);
      }
      appendOp(cb, finalstep, conf.sc.X(), conf.sc.Y(), 1);
    }
    cb.end();
    for (u32 i = 0; i < conf.steps; ++i) {
      mgr.execute(cb);
    }
    hid_t file =
        H5Fcreate(conf.output.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t sizes[2] = {conf.sc.nx, conf.sc.ny};
    hid_t space = H5Screate_simple(2, sizes, nullptr);
    hid_t c_float = H5Tcreate(H5T_COMPOUND, sizeof(c32));
    H5Tinsert(c_float, "r", 0, H5T_NATIVE_FLOAT_g);
    H5Tinsert(c_float, "i", 4, H5T_NATIVE_FLOAT_g);
    hid_t set = H5Dcreate2(file, "psir", c_float, space, H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);

    mgr.writeFromBuffer(psir, cpu_psir);
    hid_t res =
        H5Dwrite(set, c_float, H5S_ALL, space, H5P_DEFAULT, cpu_psir.data());
    if (res < 0) {
      std::cout << "Failed to write HDF5\n";
    }
    H5Fclose(file);
    deleteVkFFT(&app);
  }
  return 0;
}

// int test_first_min_max() {
//   Manager mgr(10 * 1024 * 1024);
//   constexpr u32 n_source = 64;
//   constexpr u32 n_target = n_source / 16;
//   std::vector<f32> values(n_source, 2.0);
//   /*for (u32 i = 0; i < n_source; i++) {
//     values[i] = 2.0;
//   }*/
//   values[n_source - 1] = -1000.;
//   values[n_source / 2] = -1001.;
//   values[n_source / 3] = 120.;
//   std::cout << "Copying to GPU\n";
//   MetaBuffer gpu_source = mgr.vecToBuffer(values);
//   MetaBuffer gpu_target = mgr.makeRawBuffer<f32>(n_target);
//   std::cout << "Number of elements in values: " << values.size() << '\n';
//   std::cout << "GPU source buffer size in bytes: " << gpu_source.aInfo.size
//             << '\n';
//   std::cout << "Done copying, initialising minimum reduction\n";
//   Algorithm firstminmax = mgr.makeAlgorithm<u32>(
//       "Shaders/firstminmax.spv", {}, {&gpu_source, &gpu_target}, n_target);
//   vk::CommandBuffer cb = mgr.beginRecord();
//   std::cout << "appending op\n";
//   u32 X = (n_source - 1 + WAVE_SIZE) / WAVE_SIZE;
//   std::cout << "Dispatching: " << X << " workgroups.\n"
//             << "Number of threads should be: " << X * 32 << '\n';
//   appendOp(cb, firstminmax, X, 1, 1);
//   cb.end();
//
//   std::cout << "Executing reduction\n";
//   mgr.execute(cb);
//   std::cout << "Copying from GPU\n";
//   mgr.writeFromBuffer(gpu_target, values.data(), (n_target) * 4);
//   std::cout << "Transferred values:\n";
//   for (u32 i = 0; i < (n_target); i++) {
//     std::cout << values[i] << '\n';
//   }
//   return 0;
// }
//
// int test_complete_min_max() {
//   Manager mgr(10 * 1024 * 1024);
//   constexpr u32 n_source = 32 * 64;
//   constexpr u32 n_target = n_source / 16;
//   std::vector<f32> values(n_source, 2.0);
//   /*for (u32 i = 0; i < n_source; i++) {
//     values[i] = 2.0;
//   }*/
//   values[n_source - 1] = -1000.;
//   values[n_source / 2] = -1001.;
//   values[n_source / 3] = 120.;
//   std::cout << "Copying to GPU\n";
//   MetaBuffer gpu_source = mgr.vecToBuffer(values);
//   MetaBuffer gpu_target = mgr.makeRawBuffer<f32>(n_target);
//   std::cout << "Number of elements in values: " << values.size() << '\n';
//   std::cout << "GPU source buffer size in bytes: " << gpu_source.aInfo.size
//             << '\n';
//   std::cout << "Done copying, initialising minimum reduction\n";
//   Algorithm firstminmax = mgr.makeAlgorithm(
//       "Shaders/firstminmax.spv", {}, {&gpu_source, &gpu_target}, n_target);
//   Algorithm findminmax =
//       mgr.makeAlgorithm("Shaders/minmax.spv", {}, {&gpu_target}, n_target);
//   vk::CommandBuffer cb = mgr.beginRecord();
//   std::cout << "appending op\n";
//   u32 X = (n_source - 1 + WAVE_SIZE) / WAVE_SIZE;
//   std::cout << "Dispatching: " << X << " workgroups.\n"
//             << "Number of threads should be: " << X * 32 << '\n';
//   appendOp(cb, firstminmax, X, 1, 1);
//   X = (X - 1 + WAVE_SIZE) / WAVE_SIZE;
//   while (X > 1) {
//     appendOp(cb, findminmax, X, 1, 1);
//     X = (X + WAVE_SIZE - 1) / WAVE_SIZE;
//   }
//   appendOp(cb, findminmax, 1, 1, 1);
//   cb.end();
//
//   std::cout << "Executing reduction\n";
//   mgr.execute(cb);
//   std::cout << "Copying from GPU\n";
//   mgr.writeFromBuffer(gpu_target, values.data(), (n_target) * 4);
//   std::cout << "Transferred values:\n";
//   for (u32 i = 0; i < (n_target); i++) {
//     std::cout << values[i] << '\n';
//   }
//   return 0;
// }
//
// // forget the benchmark actually, looping externally should be fine
// int test_min_max() {
//   Manager mgr(10 * 1024 * 1024);
//   u32 sharedDataSize = std::min(
//       1024u,
//       static_cast<u32>(
//           mgr.physicalDevice.getProperties().limits.maxComputeSharedMemorySize
//           / sizeof(f32)));
//   constexpr u32 n_elements = 3 * 1024 * 1024 + 32;
//   std::vector<f32> values(n_elements, 2.0);
//   /*for (u32 i = 0; i < n_elements; i++) {
//     values[i] = 2.0;
//   }*/
//   values[n_elements - 1] = -1000.;
//   values[n_elements / 2] = -1001.;
//   values[n_elements / 3] = 120.;
//   std::cout << "Copying to GPU\n";
//   MetaBuffer gpu_values = mgr.vecToBuffer(values);
//   std::cout << "Number of elements in values: " << values.size() << '\n';
//   std::cout << "GPU buffer size in bytes: " << gpu_values.aInfo.size << '\n';
//   std::cout << "Done copying, initialising minimum reduction\n";
//   size_t pushSize = 4;
//   Algorithm findminmax = mgr.makeAlgorithm<u32>("Shaders/minmax.spv", {},
//                                                 {&gpu_values}, n_elements);
//   vk::CommandBuffer cb = mgr.beginRecord();
//   std::cout << "appending op\n";
//   u32 X = (n_elements - 1 + WAVE_SIZE) / WAVE_SIZE;
//   std::cout << "Dispatching: " << X << " workgroups.\n"
//             << "Number of threads should be: " << X * 32 << '\n';
//   while (X > 1) {
//     appendOp(cb, findminmax, X, 1, 1);
//     X = (X + WAVE_SIZE - 1) / WAVE_SIZE;
//   }
//   appendOp(cb, findminmax, 1, 1, 1);
//   cb.end();
//
//   std::cout << "Executing reduction\n";
//   mgr.execute(cb);
//   std::cout << "Copying from GPU\n";
//   mgr.writeFromBuffer(gpu_values, values.data(), n_elements * 4);
//   std::cout << "First 128 values after reduction:\n";
//   for (u32 i = 0; i < 1; i++) {
//     std::cout << values[i] << '\n';
//   }
//   std::cout << values[n_elements - 1] << '\n';
//   return 0;
// }
//
// int test_graphical() {
//   auto window = create_window_sdl("Bleh", SDL_WINDOW_RESIZABLE);
//   {
//     Manager mgr(10 * 1024 * 1024, window);
//     Renderer renderer(mgr, sc.nx, sc.ny);
//     bool running = true;
//     while (running) {
//       SDL_Event event;
//       while (SDL_PollEvent(&event)) {
//         if (event.type == SDL_EVENT_QUIT ||
//             (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
//              event.window.windowID == SDL_GetWindowID(window))) {
//           running = false;
//           break;
//         }
//         if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
//           SDL_Delay(10);
//           continue;
//         }
//       }
//       renderer.drawFrame();
//     }
//     std::vector<float> lets_take_a_peek(512 * 512);
//     mgr.writeFromBuffer(renderer.value_buffer, lets_take_a_peek.data(),
//                         512 * 512);
//     std::ofstream f;
//     f.open("valuedata.csv");
//     writeCsv(f, lets_take_a_peek, 512, 512);
//     std::vector<f32> minmax_data(512 * 512 / 16);
//     mgr.writeFromBuffer(renderer.minmax_buffer, minmax_data.data(),
//                         512 * 512 / 16);
//     f.open("minmaxdata.csv");
//     writeCsv(f, minmax_data, 512, 512 / 16);
//     std::cout << window << std::endl;
//   }
//   std::cout << window << std::endl;
//   SDL_DestroyWindow(window);
//   return 0;
// }
//
// int execute_graphical() {
//   std::cout << "started\n";
//   auto window = create_window_sdl("Bleh", SDL_WINDOW_RESIZABLE);
//   {
//     Manager mgr(10 * 1024 * 1024, window);
//     Renderer renderer(mgr, sc.nx, sc.ny);
//     std::vector<c32> cpu_psir(sc.nx * sc.ny);
//     std::random_device dev;
//     std::mt19937 gen(dev());
//     std::uniform_real_distribution<f32> dis(-1e-8, 1e-8);
//     for (auto& x : cpu_psir) {
//       x = c32{dis(gen), dis(gen)};
//     }
//     constexpr f32 xstart = -48.0;
//     constexpr f32 xend = 48.0;
//     constexpr f32 dx = (xend - xstart) / sc.nx;
//     constexpr f32 kmax = M_PI / dx;
//     std::vector<c32> cpu_kProp(sc.nx * sc.ny);
//     for (u32 j = 0; j < sc.ny; j++) {
//       f32 ky = int_to_coord(j, sc.ny, -kmax, kmax);
//       for (u32 i = 0; i < sc.nx; i++) {
//         f32 kx = int_to_coord(i, sc.nx, -kmax, kmax);
//         cpu_kProp[j * sc.nx + i] = std::exp(
//             c32{0.0, -0.5f * hbar * sc.dt * (square(kx) + square(ky)) /
//             0.32f});
//       }
//     }
//     MetaBuffer psir = mgr.vecToBuffer(cpu_psir);
//     MetaBuffer oldPsir = mgr.makeRawBuffer<c32>(sc.nx * sc.ny);
//     MetaBuffer psik = mgr.makeRawBuffer<c32>(sc.nx * sc.ny);
//     MetaBuffer nR = mgr.makeRawBuffer<f32>(sc.nx * sc.ny);
//     mgr.defaultInitBuffer<f32>(nR, sc.nx * sc.ny);
//     MetaBuffer kTimeEvo = mgr.vecToBuffer(cpu_kProp);
//     std::vector<f32> cpu_pump(sc.nx * sc.ny);
//     for (u32 j = 0; j < sc.ny; j++) {
//       f32 y = int_to_coord(j, sc.ny, xstart, xend);
//       for (u32 i = 0; i < sc.nx; i++) {
//         f32 x = int_to_coord(i, sc.nx, xstart, xend);
//         std::cout << x << ' ';
//         cpu_pump[j * sc.nx + i] = 10.0 * pumpProfile(x, y, 1.8, 4.3, 0.9);
//       }
//     }
//     MetaBuffer pump = mgr.vecToBuffer(cpu_pump);
//
//     Algorithm rstep = mgr.makeAlgorithm("Shaders/rstep.spv", {},
//                                         {&psir, &oldPsir, &nR, &pump}, sc);
//     Algorithm kstep =
//         mgr.makeAlgorithm("Shaders/kstep.spv", {}, {&psik, &kTimeEvo}, sc);
//     Algorithm finalstep = mgr.makeAlgorithm("Shaders/finalstep.spv", {},
//                                             {&psir, &oldPsir, &nR, &pump},
//                                             sc);
//     Algorithm transfer = mgr.makeAlgorithm("Shaders/transferandsquare.spv",
//     {},
//                                            {&psir, &renderer.value_buffer},
//                                            sc);
//     VkFFTConfiguration conf{};
//     conf.device = pcast<VkDevice>(&mgr.device);
//     conf.FFTdim = 2;
//     conf.size[0] = sc.nx;
//     conf.size[1] = sc.ny;
//     conf.queue = pcast<VkQueue>(&mgr.queue);
//     conf.fence = pcast<VkFence>(&mgr.fence);
//     conf.commandPool = pcast<VkCommandPool>(&mgr.commandPool);
//     conf.physicalDevice = pcast<VkPhysicalDevice>(&mgr.physicalDevice);
//     conf.buffer = pcast<VkBuffer>(&psik.buffer);
//     conf.isInputFormatted = true;
//     conf.inputBuffer = pcast<VkBuffer>(&psir.buffer);
//     conf.bufferSize = &psik.aInfo.size;
//     conf.inputBufferSize = &psir.aInfo.size;
//     conf.inverseReturnToInputBuffer = true;
//     conf.normalize = true;
//     VkFFTApplication app{};
//     auto resFFT = initializeVkFFT(&app, conf);
//     if (resFFT != VKFFT_SUCCESS) {
//       std::cout << resFFT << '\n';
//       exit(1);
//     }
//     auto cb = mgr.beginRecord();
//     VkFFTLaunchParams launchParams{};
//     launchParams.commandBuffer = pcast<VkCommandBuffer>(&cb);
//     for (int i = 0; i < 100; i++) {
//       appendOp(cb, rstep, sc.X(), sc.Y(), 1);
//       resFFT = VkFFTAppend(&app, -1, &launchParams);
//       if (resFFT != VKFFT_SUCCESS) {
//         std::cout << resFFT << '\n';
//         exit(1);
//       }
//       appendOp(cb, kstep, sc.X(), sc.Y(), 1);
//       resFFT = VkFFTAppend(&app, 1, &launchParams);
//       if (resFFT != VKFFT_SUCCESS) {
//         std::cout << resFFT << '\n';
//         exit(1);
//       }
//       appendOp(cb, finalstep, sc.X(), sc.Y(), 1);
//     }
//     cb.end();
//     auto xferCommand = mgr.beginRecord();
//     appendOp(xferCommand, transfer, sc.X(), sc.Y(), 1);
//     xferCommand.end();
//
//     mgr.execute(cb);
//     mgr.writeFromBuffer(psir, cpu_psir);
//     for (const auto& x : cpu_psir) {
//       std::cout << x << ' ';
//     }
//     bool running = true;
//     while (running) {
//       SDL_Event event;
//       while (SDL_PollEvent(&event)) {
//         if (event.type == SDL_EVENT_QUIT ||
//             (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
//              event.window.windowID == SDL_GetWindowID(window))) {
//           running = false;
//           break;
//         }
//         if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
//           SDL_Delay(10);
//           continue;
//         }
//       }
//       mgr.execute(cb);
//       mgr.execute(xferCommand);
//       renderer.drawFrame();
//     }
//     mgr.writeFromBuffer(psir, cpu_psir);
//     for (const auto& x : cpu_psir) {
//       std::cout << x << ' ';
//     }
//
//     deleteVkFFT(&app);
//   }
//
//   SDL_DestroyWindow(window);
//   return 0;
// }
