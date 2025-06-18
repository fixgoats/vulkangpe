RWStructuredBuffer<float> minmax;

[shader("compute")]
[numthreads(64, 1, 1)]
void findminloop(uint3 threadId: SV_DispatchThreadID, uniform uint stride) {
  uint i = threadId.x;
  uint count = 0;
  uint sizeT = 0;
  minmax.GetDimensions(count, sizeT);
  for (uint local_stride = stride; local_stride > 1; local_stride = (local_stride + 1) / 2) {
    if ((i < local_stride) && (i + local_stride < count)) {
      minmax[i] = min(minmax[i], minmax[i + stride]);
    }
  }
  minmax[0] = min(minmax[0], minmax[i + stride]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void findminsingle(uint3 threadId: SV_DispatchThreadID, uniform uint stride) {
  uint i = threadId.x;
  uint N = 0;
  uint _ = 0;
  minmax.GetDimensions(N, _);
  if (i + stride < N) {
    minmax[i] = min(minmax[i], minmax[i + stride]);
  }
}
