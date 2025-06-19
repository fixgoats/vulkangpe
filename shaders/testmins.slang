RWStructuredBuffer<float> minmax;
groupshared float sdata[1024];

[shader("compute")]
[numthreads(64, 1, 1)]
void findminloop(uint3 threadId: SV_DispatchThreadID, uniform uint stride) {
  uint i = threadId.x;
  uint count = 0;
  uint sizeT = 0;
  minmax.GetDimensions(count, sizeT);
  uint iter = 0;
  for (uint local_stride = stride; local_stride > 1; local_stride = (local_stride + 1) / 2) {
    if ((i < local_stride) && (i + local_stride < count)) {
      minmax[i] = min(minmax[i], minmax[i + local_stride]);
    }
  }
  minmax[0] = min(minmax[0], minmax[1]);
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void findminsingle(uint tId: SV_DispatchThreadID, uint gId: SV_GroupThreadID, uniform uint stride) {
  sdata[gId] = minmax[tId];
  uint N = 0;
  uint _ = 0;
  minmax.GetDimensions(N, _);
  if (i + stride < N) {
    sdata[gId] = min(sdata[gId], minmax[gId + stride]);
  }
}
