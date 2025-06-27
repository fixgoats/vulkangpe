cmake --build build --parallel
slangc shaders/minmax.slang -profile glsl_450+spvGroupNonUniformArithmetic -target spirv -o build/Shaders/minmax.spv -entry findminmax
slangc shaders/firstminmax.slang -profile glsl_450+spvGroupNonUniformArithmetic -target spirv -o build/Shaders/firstminmax.spv -entry firstminmax
slangc shaders/colormap.slang -profile glsl_450 -target spirv -o build/Shaders/colormap.spv -entry writecolor
slangc shaders/quad.slang -profile glsl_450 -target spirv -o build/Shaders/quad.vert.spv -entry vertexMain
slangc shaders/quad.slang -profile glsl_450 -target spirv -o build/Shaders/quad.frag.spv -entry fragmentMain
