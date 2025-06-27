StructuredBuffer<float> source;
RWStructuredBuffer<float> minmax;

[[SpecializationConstant]]
const uint NELEMENTS = 2;

[shader("compute")]
[numthreads(32, 1, 1)]
void firstminmax(uint tId: SV_DispatchThreadID, uint gId: SV_GroupID) {
  float x = source[tId];
  minmax[gId] = WaveActiveMin(x);
  minmax[NELEMENTS - gId - 1] = WaveActiveMax(x);
}
