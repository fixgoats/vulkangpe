#include <complex.slang>

RWStructuredBuffer<float2> psir;
RWStructuredBuffer<float2> oldpsir;
RWStructuredBuffer<float> nR;
StructuredBuffer<float> pump;

[SpecializationConstant]
const uint XGROUPSIZE = 1;
[SpecializationConstant]
const float alpha = 4e6;
[SpecializationConstant]
const float gammalp = 2e6;
[SpecializationConstant]
const float Gamma = 1e6;
[SpecializationConstant]
const float G = 2e6;
[SpecializationConstant]
const float R = 100.015;
[SpecializationConstant]
const float eta = 2.e6;
[SpecializationConstant]
 const float dt = 1e6;

layout(local_size_x_id = 2, local_size_y = 1, local_size_z = 1) in;


float2 halfRTimeEvo(float2 psi, float nr, float p) {
    return c_exp(float2(0.25 * dt * (R * nr - gammalp), -0.5 * dt * ((G * eta / Gamma) * p + alpha * dot(psi, psi) + G * nr)));
}

[shader("compute")]
[numthreads(XGROUPSIZE, 1, 1)]
void rstep(uint3 threadId: SV_DispatchThreadID) {
  uint i = threadId.x;
  float p = pump[i];
  float2 psi = psir[i];
  psir[i] = c_mul(halfRTimeEvo(psi, nR[i], p), psi);
  nR[i] = RK4Step(nR[i], p, oldpsir[i], psir[i]);
}
