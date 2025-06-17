// rstep.slang
import complex;

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

float2 halfRTimeEvo(float2 psi, float nr, float p) {
  return c_exp(float2(
      0.25 * dt * (R * nr - gammalp),
      -0.5 * dt * ((G * eta / Gamma) * p + alpha * dot(psi, psi) + G * nr)));
}

[shader("compute")]
[numthreads(XGROUPSIZE, 1, 1)]
void rstep(uint3 threadId: SV_DispatchThreadID) {
  uint i = threadId.x;
  oldpsir[i] = psir[i];
  psir[i] = c_mul(halfRTimeEvo(psir[i], nR[i], pump[i]), psir[i]);
}

float f(float y, float p, float2 psi) {
  return -(Gamma + R * dot(psi, psi)) * y + p;
}

float RK4Step(float y, float p, float2 psi0, float2 psi) {
  float k1 = f(y, p, psi0);
  float k2 = f(y + (0.5 * dt) * k1, p, 0.5 * (psi0 + psi));
  float k3 = f(y + (0.5 * dt) * k2, p, 0.5 * (psi0 + psi));
  float k4 = f(y + dt * k3, p, psi);
  return y + dt * (1. / 6.) * (k1 + (2 * k2) + (2 * k3) + k4);
}

/*struct Exciton : ISystem, IVec {
  float n;
  Exciton scalar_mul(float a) { return { a * this.n }; }
  Exciton system() { return { -(Gamma + R) * n }; }
}
extension float : ISystem, IVec {
  float scalar_mul(float b) { return this * b; }
  float system() { return -(Gamma + R) * this; }
}*/

[shader("compute")]
[numthreads(XGROUPSIZE, 1, 1)]
void finalstep(uint3 threadId: SV_DispatchThreadID) {
  uint i = threadId.x;
  float p = pump[i];
  float3 b = float3(1., 2., 3.) * float3(3., 2., 4.);
  psir[i] = c_mul(halfRTimeEvo(psir[i], nR[i], p), psir[i]);
  nR[i] = RK4Step(nR[i], dt, p, oldpsir[i], psir[i]);
}

