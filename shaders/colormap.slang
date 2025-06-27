RWTexture2D<float4> field;

StructuredBuffer<float4> colors;
// min is at minmax[0], max at minmax[NMINMAXVALUES-1]
StructuredBuffer<float> minmax;
StructuredBuffer<float> values;
[[SpecializationConstant]]
const uint NMINMAXVALUES;

[shader("compute")]
[numthreads(8, 8, 1)]
void writecolor(uint3 threadId: SV_DispatchThreadID, ) {
  uint width = 0;
  uint height = 0;
  field.GetDimensions(width, height);
  uint vidx = threadId.y * width + threadId.x;
  uint coloridx = uint(256 * (values[vidx] - minmax[0]) /
                       (minmax[NMINMAXVALUES - 1] - minmax[0]));
  int2 coord = int2(threadId.xy);
  field[coord] = colors[coloridx];
}
