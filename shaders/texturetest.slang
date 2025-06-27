RWTexture2D<float4> image;
ConstantBuffer<float4[256]> colors;

[shader("compute")]
[numthreads(8, 8, 1)]
void write(uint3 threadId: SV_DispatchThreadID) {
  int2 coord = int2(threadId.xy);
  image[coord] = colors[1];
}
