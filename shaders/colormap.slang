WTexture2D colormap;
ConstantBuffer<float4> colors[256];
// min is at minmax[0], max at minmax[1]
RStructuredBuffer<float> minmax;
RStructuredBuffer<float> values;

[shader("compute")]
[numthreads(8, 8, 1)]
void writecolor(uint3 threadId: SV_DispatchThreadID) {
    uint coloridx = uint(256 * (values[uidx] - minmax[0]) / (minmax[1] - minmax[0]
    colormap[threadId.xy].Load(colors[coloridx]);
}
