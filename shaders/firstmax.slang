StructuredBuffer<float> values;
RWStructuredBuffer<float> minmax;


[shader("compute")]
[numthreads(64, 1, 1)]
void firststep(uint3 threadId: SV_DispatchThreadID) {
    uint i = threadId.x;
    uint N = values.length();
    if (2 * i + 1 < N) {
        if (values[2 * i] <= values[2 * i + 1]) {
            minmax[i] = values[2 * i];
            minmax[N / 2]
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void firstmax(uint3 threadId: SV_DispatchThreadID) {
    uint i = threadId.x;
    if (2 * i + 1 < data.length()) {
        minmax[i+1] = max(data[2 * i], data[2*i+1])
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void findminloop(uint3 threadId: SV_DispatchThreadID, uniform stride: uint) {
  uint i = threadId.x;
  uint local_stride = stride;
  for (uint local_stride = stride; local_stride > 1; local_stride = (local_stride + 1) / 2) {
    if (idx < stride && idx + stride < minmax.length()) {
      minmax[i] = min(minmax[idx], minmax[idx + stride]);
    }
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void findminsingle(uint3 threadId: SV_DispatchThreadID, uniform stride: uint) {
  uint idx = threadId.x;
  if (idx + stride < minmax.length()) {
    minmax[idx] = min(minmax[idx], minmax[idx + stride]);
  }
}
