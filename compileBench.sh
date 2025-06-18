cmake --build build --parallel
slangc shaders/testmins.slang -profile glsl_450 -target spirv -o build/Shaders/findminloop.spv -entry findminloop
slangc shaders/testmins.slang -profile glsl_450 -target spirv -o build/Shaders/findminsingle.spv -entry findminsingle
