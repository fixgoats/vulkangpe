#pragma once
#include <cstdint>

template <typename T>
constexpr uint32_t euclid_mod(T a, uint32_t b) {
  assert(b != 0);
  return a % b;
}

template <typename T>
constexpr T square(T x) {
  return x * x;
}
constexpr uint32_t fftshiftidx(uint32_t i, uint32_t n) {
  return euclid_mod(i + (n + 1) / 2, n);
}

template <typename T>
void leftRotate(std::vector<T>& arr, uint32_t d) {
  auto n = arr.size();
  d = d % n; // To handle case when d >= n

  // Reverse the first d elements
  std::reverse(arr.begin(), arr.begin() + d);

  // Reverse the remaining elements
  std::reverse(arr.begin() + d, arr.end());

  // Reverse the whole array
  std::reverse(arr.begin(), arr.end());
}

template <typename T>
void fftshift(std::vector<T>& arr) {
  auto n = arr.size();
  uint32_t d = (n + 1) % 2;
  leftRotate(arr, d);
}

constexpr float pumpProfile(float x, float y, float L, float r, float beta) {
  return square(square(L)) /
         (square(x * x + beta * y * y - r * r) + square(square(L)));
}
