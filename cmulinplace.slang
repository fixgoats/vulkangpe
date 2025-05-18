#include <complex.slang>
[SpecializationConstant]
const uint XGROUPSIZE = 1;

StructuredBuffer<float2> a;
RWStructuredBuffer<float2> b;

[shader("compute")]
[numthreads(XGROUPSIZE, 1, 1)]
void cmulMain(uint3 threadId: SV_DispatchThreadID) {
    uint i = threadId.x;
    b[i] = c_mul(b[i], a[i]);
}
