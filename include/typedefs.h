#pragma once
#include <complex>
#include <cstdint>

typedef std::complex<float> c32;
typedef std::complex<double> c64;

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int64_t s64;
typedef int32_t s32;
typedef int16_t s16;
typedef int8_t s8;
typedef float f32;
typedef double f64;

template <class T>
struct vec4 {
  T x, y, z, w;
};

template <class T>
struct vec3 {
  T x;
  T y;
  T z;
};

template <class T>
struct vec2 {
  T x, y;
};
