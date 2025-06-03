#pragma once
float2 c_mul(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 c_exp(float2 u) { return exp(u.x) * float2(cos(u.y), sin(u.y)); }

float2 conj(float2 a) { return float2(a.x, -a.y); }

float2 c_inv(float2 a) { return conj(a) / dot(a, a); }

float2 c_div(float2 a, float b) { return a * c_inv(b); }

float2 c_cos(float2 a) {
  return float2(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y));
}

float2 c_sin(float2 a) {
  return float2(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y));
}

float2 c_polar(float2 a) { return { length(a), atan2(a.y, a.x) }; }
