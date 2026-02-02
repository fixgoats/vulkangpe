#pragma once
#include "betterexc.hpp"
#include "typedefs.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>

constexpr f32 hbar = 6.582119569e-1;
constexpr f32 muB = 5.7883818060e-2;
constexpr f32 echarge = 1e3;

using std::bit_cast;

template <typename T>
constexpr u32 euclid_mod(T a, u32 b) {
  assert(b != 0);
  return a % b;
}

template <class T>
constexpr auto numfmt(T x) {
  if constexpr (std::is_same_v<T, c32> or std::is_same_v<T, c64>) {
    return std::format("({}+{}j)", x.real(), x.imag());
  } else {
    return std::format("{}", x);
  }
}

template <class T>
void writeCsv(std::ofstream& of, const std::vector<T>& v, u32 nColumns,
              u32 nRows = 1, u32 stride = 1, u32 offset = 0,
              const std::vector<std::string>& heading = {}) {
  if (offset + stride * nColumns * nRows < v.size()) {
    runtime_exc("There aren't this many elements in the vector.");
  }
  std::string out;
  if (heading.size()) {
    for (const auto& h : heading) {
      of << h << ' ';
    }
    of << '\n';
  }
  for (u32 j = 0; j < nRows; j++) {
    for (u32 i = 0; i < nColumns; i += stride) {
      of << v[j * nColumns * stride + i + offset] << ' ';
    }
    of << '\n';
  }
  of.close();
}

template <>
void writeCsv<c32>(std::ofstream& of, const std::vector<c32>& v, u32 nColumns,
                   u32 nRows, u32 stride, u32 offset,
                   const std::vector<std::string>& heading);
template <>
void writeCsv<c64>(std::ofstream& of, const std::vector<c64>& v, u32 nColumns,
                   u32 nRows, u32 stride, u32 offset,
                   const std::vector<std::string>& heading);

template <class T>
void writeBinary(std::string filename, std::span<T> span) {
  std::ofstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    runtime_exc("Failed to open file: {}", filename);
  }

  file.write(reinterpret_cast<char*>(span.data()), span.size() * sizeof(T));
  file.close();
}

static inline u32 uintlog2(const u32 x) {
  uint32_t y;
  asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
  return y;
}

/*template <class T, u32 C, u32 R>
struct small_mat {
  std::array<T, R * C> buffer;

  // it seems like in at least some cases memcpying an array of arrays with the
  // total size does give you a flattened array, but I don't think this is
  // something I can count on.
  small_mat(const T (&ll)[R][C]) {
    for (size_t j = 0; j < R; j++) {
      for (size_t i = 0; i < C; i++) {
        buffer[j * C + i] = ll[j][i];
      }
    }
  }

  constexpr u32 X() { return C; }
  constexpr u32 Y() { return R; }
};

// For when you need a more flexible matrix.
template <class T>
struct mat {
  static_assert(std::is_same_v<T, float> or std::is_same_v<T, c32> or
                    std::is_same_v<T, double> or std::is_same_v<T, c64>,
                "Type must be one of: float, c32, double or c64");
  std::array<u32, 2> dims;
  std::vector<T> buffer;

  template <u32 C, u32 R>
  mat(const T (&ll)[C][R]) {
    dims = {C, R};
    buffer.resize(R * C);
    for (size_t j = 0; j < R; j++) {
      for (size_t i = 0; i < C; i++) {
        buffer[j * C + i] = ll[j][i];
      }
    }
  }
  mat() {
    dims = {0, 0};
    buffer = {};
  }
  mat(u32 c, u32 r) {
    dims = {c, r};
    buffer.resize(c * r);
  }

  constexpr u32 X() { return dims[0]; }
  constexpr u32 Y() { return dims[1]; }
  constexpr T* data() { return buffer.data(); }
  constexpr size_t size() { return buffer.size(); }

  void savetxt(std::string fname) {
    std::ofstream file(fname, std::ios::binary);
    if (!file.is_open()) {
      throw runtime_exc{"Couldn't open file: {}", fname};
    }

    file.write(reinterpret_cast<const char*>(dims.data()), 8);
    file.write(reinterpret_cast<const char*>(buffer.data()),
               buffer.size() * sizeof(T));
    file.close();
  }

  mat(const std::vector<char>& buf) {
    memcpy(dims.data(), buf.data(), 8);
    buffer.resize(dims[0] * dims[1]);
    memcpy(buffer.data(), buf.data() + 8, buf.size() - 8);
  }

  mat(const std::vector<char>&& buf) {
    memcpy(dims.data(), buf.data(), 8);
    buffer.resize(dims[0] * dims[1]);
    memcpy(buffer.data(), buf.data() + 8, buf.size() - 8);
  }
};

template <class T, u32 S, u32 C, u32 R>
struct small_arr3 {
  // small_* is just the array, dimensions are compile time-constants, hence
  // no specific serialization, just reading/writing the flat data.
  std::array<T, S * C * R> data;

  small_arr3(const T (&ll)[S][R][C]) {
    for (size_t k = 0; k < S; k++) {
      for (size_t j = 0; j < R; j++) {
        for (size_t i = 0; i < C; i++) {
          data[(k * R + j) * C + i] = ll[k][j][i];
        }
      }
    }
  }
  constexpr u32 X() { return R; }
  constexpr u32 Y() { return C; }
  constexpr u32 Z() { return S; }
};

template <class T>
struct arr3 {
  std::array<u32, 3> dims;
  std::vector<T> data{};

  template <u32 C, u32 R, u32 S>
  constexpr arr3(const T (&ll)[S][R][C]) {
    std::cout << "slices: " << S << '\n';
    std::cout << "rows: " << R << '\n';
    std::cout << "columns: " << C << '\n';
    data.resize(S * C * R);
    for (size_t k = 0; k < S; k++) {
      for (size_t j = 0; j < R; j++) {
        for (size_t i = 0; i < C; i++) {
          data.push_back(ll[k][j][i]);
        }
      }
    }
  }

  constexpr arr3(u32 rows, u32 cols, u32 slices) {
    dims = {rows, cols, slices};
    std::vector<T> data(slices * cols * rows);
  }

  constexpr u32 X() { return dims[0]; }
  constexpr u32 Y() { return dims[1]; }
  constexpr u32 Z() { return dims[2]; }

  void save(std::string fname) {
    std::ofstream file(fname, std::ios::binary);
    if (!file.is_open()) {
      throw runtime_exc{"Couldn't open file: {}", fname};
    }

    file.write(reinterpret_cast<const char*>(dims.data()), 12);
    file.write(reinterpret_cast<const char*>(data.data()),
               data.size() * sizeof(T));
    file.close();
  }

  // It may be in some way more performant and flexible to take a bare pointer
  // rather than a vector reference, but a reference is guaranteed to be
  // non-null and also allows capturing temporals:
  // https://stackoverflow.com/a/52255382/6461823
  arr3(const std::vector<char>& buf) {
    memcpy(dims.data(), buf.data(), 12);
    data.resize(dims[0] * dims[1] * dims[2]);
    memcpy(data.data(), buf.data() + 12, buf.size() - 12);
  }

  arr3(const std::vector<char>&& buf) {
    memcpy(dims.data(), buf.data(), 12);
    data.resize(dims[0] * dims[1] * dims[2]);
    memcpy(data.data(), buf.data() + 12, buf.size() - 12);
  }
};*/

template <typename T>
constexpr T square(T x) {
  return x * x;
}
constexpr u32 fftshiftidx(u32 i, u32 n) {
  return euclid_mod(i + (n + 1) / 2, n);
}

template <typename T>
void leftRotate(std::vector<T>& arr, u32 d) {
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
  u32 d = (n + 1) % 2;
  leftRotate(arr, d);
}

constexpr float annularProfile(float x, float y, float L, float r, float beta) {
  return square(square(L)) /
         (square(x * x + beta * y * y - r * r) + square(square(L)));
}

template <typename T>
u8 mapToColor(T v, T min, T max) {
  return static_cast<u8>(256 * (v - min) / (max - min));
}

template <typename InputIt, typename OutputIt>
void colorMap(InputIt it, InputIt end, OutputIt out) {
  const auto max = *std::max_element(it, end);
  const auto min = *std::min_element(it, end);
  std::transform(it, end, out, [&](auto x) { return mapToColor(x, min, max); });
}

template <typename InputIt>
std::vector<u8> colorMapVec(InputIt it, InputIt end) {
  std::vector<u8> out(end - it);
  colorMap(it, end, out.begin());
  return out;
}
