#!/bin/bash
cmake --build build --parallel
glslangValidator -V shaders/rstep.comp -o build/rstep.spv
glslangValidator -V shaders/kstep.comp -o build/kstep.spv
glslangValidator -V shaders/finalstep.comp -o build/finalstep.spv
glslangValidator -V shaders/TexturedQuad.vert -o build/Shaders/TexturedQuad.vert.spv
glslangValidator -V shaders/TexturedQuad.frag -o build/Shaders/TexturedQuad.frag.spv
glslangValidator -V shaders/FillTexture.comp -o build/Shaders/FillTexture.comp.spv
