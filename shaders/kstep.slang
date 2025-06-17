import complex;

RWStructuredBuffer<float2> psik;
RStructuredBuffer<float2> kTimeEvo;

[shader("compute")]
[numthreads(64, 1, 1)]
void kstep(uint3 threadId: SV_DispatchThreadID) {
    uint i = threadId.x;
    psik[i] = c_mul(kTimeEvo[i], psik[i]);
}
