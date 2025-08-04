[SpecializationConstant]
const uint NELEMENTSX = 2;
[SpecializationConstant]
const uint XGROUPSIZE = 1;

StructuredBuffer<float2> src;
RWStructuredBuffer<float> dest;

[shader("compute")]
[numthreads(XGROUPSIZE, 1, 1)]
void transferAndSquare(uint threadId: SV_DispatchThreadID) {
  uint i = threadId;
  dest[i] = dot(src[i]);
}
