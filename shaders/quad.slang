struct VSInput {
  [[vk::location(0)]]
  float2 position;
  [[vk::location(1)]]
  float2 uv;
}

struct VSOutput {
  float4 position : SV_Position;
  [[vk::location(0)]]
  float2 uv;
}

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
  VSOutput output;
  output.position = float4(input.position, 0.0, 1.0);
  output.uv = input.uv;
  return output;
}

Sampler2D sampler;

[shader("fragment")]
float4 fragmentMain(VSOutput input) { return sampler.Sample(input.uv); }
