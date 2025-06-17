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
void findmin(uint3 threadId: SV_DispatchThreadID, uniform stride: uint) {
  uint i = threadId.x;
  if (idx < stride && idx + stride < values.length()) {
    minmax[i] = max(minmax[idx], minmax[idx + stride]);
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void findmax(uint3 threadId: SV_DispatchThreadID, uniform stride: uint) {
  uint idx = stride * threadId.x + 1;
  if (idx + stride / 2 < maxima.length()) {
    maxima[idx] =
        max(maxima[idx], maxima[idx + stride / 2]);
  }
}
