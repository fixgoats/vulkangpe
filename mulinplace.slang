[SpecializationConstant]
const uint XGROUPSIZE = 1;

StructuredBuffer<float> a;
RWStructuredBuffer<float> b;

[shader("compute")]
[numthreads(XGROUPSIZE, 1, 1)]
void mulMain(uint3 threadId: SV_DispatchThreadID) {
    uint i = threadId.x;
    b[i] = b[i] * a[i];
}
