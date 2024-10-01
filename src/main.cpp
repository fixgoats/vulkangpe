#include "hack.hpp"
#include <QSurface>
#include <QWindow>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <qapplication.h>
#include <qwidget.h>
#include <qwindow.h>
#include <random>

#include "vkFFT.h"
#include <QVulkanInstance>
#include <complex>
#include <cxxopts.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "GPEsim.hpp"
#include "mainwindow.h"
#include "mathhelpers.hpp"
#include "vkFFT/vkFFT_AppManagement/vkFFT_DeleteApp.h"
#include "vkhelpers.hpp"

void updateImage(cv::Mat& img, cv::Mat& out_img, VulkanApp& GPEsim) {
  std::cout << "Running updateImage\n";
  GPEsim.runSim(1);
  auto psiR = GPEsim.outputBuffer<c32>(0);
  std::vector<float> a(GPEsim.params.elementsTotal());
  std::transform(psiR.begin(), psiR.end(), a.begin(),
                 [](c32 x) { return square(x.real()) + square(x.imag()); });
  auto max = *std::max_element(a.begin(), a.end());
  std::cout << "{\'psisqmax\': " << max << ", ";
  auto maxinv = 1 / max;
  std::transform(a.begin(), a.end(), img.begin<char>(),
                 [&](float x) { return static_cast<char>(x * maxinv * 256); });
  psiR = GPEsim.outputBuffer<c32>(1);
  std::transform(psiR.begin(), psiR.end(), a.begin(),
                 [](c32 x) { return square(x.real()) + square(x.imag()); });
  max = *std::max_element(a.begin(), a.end());
  std::cout << "\'kTimeEvoMax\': " << max << ", ";
  a = GPEsim.outputBuffer<float>(3);
  max = *std::max_element(a.begin(), a.end());
  std::cout << "\'nRMax\': " << max << "}" << std::endl;
  cv::applyColorMap(img, out_img, cv::COLORMAP_VIRIDIS);
}

int main(int argc, char* argv[]) {
  VulkanApp GPEsim({nElementsX, nElementsY, xGroupSize, yGroupSize, alpha,
                    gammalp, Gamma, G, R, eta, dt});
  GPEsim.initBuffers();
  QApplication app(argc, argv);
  MainWindow w;
  w.show();
  QVulkanInstance instance;
  instance.setVkInstance(GPEsim.instance);
  instance.create();
  QWindow window;
  window.setSurfaceType(QSurface::VulkanSurface);
  window.setVulkanInstance(&instance);
  QWidget* wrapper = QWidget::createWindowContainer(&window);
  VkSurfaceKHR surface = QVulkanInstance::surfaceForWindow(&window);
  return app.exec();
}

/*int main(int argc, char* argv[]) {
  cxxopts::Options options(appName,
                           "Vulkan simulation of Gross-Pitaevskii equation");
  options.add_options()("n,nsteps", "Number of steps to take",
                        cxxopts::value<uint32_t>())(
      "d,debug", "step through simulation one step at a time",
      cxxopts::value<bool>())("k,kout", "look at time evolution operator",
                              cxxopts::value<bool>());
  auto result = options.parse(argc, argv);
  if (result.count("n")) {
    auto start = std::chrono::high_resolution_clock::now();
    VulkanApp GPEsim({nElementsX, nElementsY, xGroupSize, yGroupSize, alpha,
                      gammalp, Gamma, G, R, eta, dt});
    std::cout << "Initialized GPE fine\n";
    GPEsim.initBuffers();
    std::cout << "Uploaded data\n";
    auto n = result["n"].as<uint32_t>();
    cv::Mat img(nElementsX, nElementsY, CV_8UC1);
    cv::Mat out_img(nElementsX, nElementsY, CV_8UC3);
    auto psiR = GPEsim.outputBuffer<c32>(0);
    std::vector<float> a(GPEsim.params.elementsTotal());
    std::transform(psiR.begin(), psiR.end(), a.begin(),
                   [](c32 x) { return square(x.real()) + square(x.imag()); });
    auto max = *std::max_element(a.begin(), a.end());
    std::cout << max << '\n';
    GPEsim.runSim(n);
    psiR = GPEsim.outputBuffer<c32>(0);
    std::transform(psiR.begin(), psiR.end(), a.begin(),
                   [](c32 x) { return square(x.real()) + square(x.imag()); });
    max = *std::max_element(a.begin(), a.end());
    auto maxinv = 1 / max;
    std::transform(a.begin(), a.end(), img.begin<char>(), [&](float x) {
      return static_cast<char>(x * maxinv * 256);
    });
    cv::applyColorMap(img, out_img, cv::COLORMAP_VIRIDIS);
    cv::imshow("Display window", out_img);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << '\n';
    int k = cv::waitKey(0);

  } else if (result.count("d")) {
    VulkanApp GPEsim({nElementsX, nElementsY, xGroupSize, yGroupSize, alpha,
                      gammalp, Gamma, G, R, eta, dt});
    std::cout << "Initialized GPE fine\n";
    GPEsim.initBuffers();
    std::cout << "Uploaded data\n";
    cv::Mat img(nElementsX, nElementsY, CV_8UC1);
    cv::Mat out_img(nElementsX, nElementsY, CV_8UC3);
    auto psiR = GPEsim.outputBuffer<c32>(0);
    std::vector<float> a(GPEsim.params.elementsTotal());
    std::transform(psiR.begin(), psiR.end(), a.begin(),
                   [](c32 x) { return square(x.real()) + square(x.imag()); });
    auto max = *std::max_element(a.begin(), a.end());
    std::cout << max << '\n';
    auto maxinv = 1 / max;
    std::transform(a.begin(), a.end(), img.begin<char>(), [&](float x) {
      return static_cast<char>(x * maxinv * 256);
    });
    cv::applyColorMap(img, out_img, cv::COLORMAP_VIRIDIS);
    cv::imshow("Display window", out_img);
    int k = cv::waitKey(0);

    while (k == 'n') {
      updateImage(img, out_img, GPEsim);
      cv::imshow("Display window", out_img);
      k = cv::waitKey(0);
    }
    return 0;
  } else if (result.count("k")) {
    std::vector<c32> vec(nElementsX * nElementsY);
    for (uint32_t j = 0; j < nElementsY; j++) {
      float kY = (float)fftshiftidx(j, nElementsY) * dKy + startKy;
      for (uint32_t i = 0; i < nElementsX; i++) {
        float kX = (float)fftshiftidx(i, nElementsX) * dKx + startKx;
        vec[j * nElementsX + i] =
            std::exp(c32{0., -(0.5f * hbar * dt / m) * (kY * kY + kX * kX)});
      }
    }
    cv::Mat img(nElementsX, nElementsY, CV_8UC1);
    cv::Mat out_img(nElementsX, nElementsY, CV_8UC3);
    std::vector<float> a(nElementsY * nElementsX);
    std::transform(vec.begin(), vec.end(), a.begin(),
                   [](c32 x) { return x.real(); });
    auto max = *std::max_element(a.begin(), a.end());
    auto min = *std::min_element(a.begin(), a.end());
    auto normfactor = 1 / (max - min);
    std::transform(a.begin(), a.end(), img.begin<char>(), [&](float x) {
      return static_cast<char>((x - min) * normfactor * 256.);
    });
    cv::applyColorMap(img, out_img, cv::COLORMAP_VIRIDIS);
    cv::imshow("Display window", out_img);
    int k = cv::waitKey(0);
    if (k == 'q') {
      return 0;
    }
  } else {
    throw std::runtime_error("gib n\n");
  }
}*/
