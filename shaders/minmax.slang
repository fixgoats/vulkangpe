RWStructuredBuffer<float> minmax;

// [[SpecializationConstant]]
// const uint SHARED_DATA_SIZE = 512;
[[SpecializationConstant]]
const uint NELEMENTS = 2;

[shader("compute")]
[numthreads(32, 1, 1)]
void findminmax(uint tId: SV_DispatchThreadID, uint gId: SV_GroupID) {
  float lesser = minmax[tId];
  float greater = minmax[NELEMENTS - tId - 1];
  minmax[gId] = WaveActiveMin(lesser);
  minmax[NELEMENTS - gId - 1] = WaveActiveMax(greater);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void findminchecked(uint tId: SV_DispatchThreadID, uint gId: SV_GroupID) {
  if (tId < NELEMENTS) {
    float lesser = minmax[tId];
    float greater = minmax[NELEMENTS - tId - 1];
    GroupMemoryBarrierWithGroupSync();
    minmax[gId] = WaveActiveMin(lesser);
    minmax[NELEMENTS - gId - 1] = WaveActiveMax(greater);
    GroupMemoryBarrierWithGroupSync();
  }
}
