#!/bin/bash
cmake --build build --parallel
glslangValidator -V shaders/rstep.comp -o build/Shaders/rstep.spv
glslangValidator -V shaders/kstep.comp -o build/Shaders/kstep.spv
glslangValidator -V shaders/finalstep.comp -o build/Shaders/finalstep.spv
glslangValidator -V shaders/TexturedQuad.vert -o build/Shaders/TexturedQuad.vert.spv
glslangValidator -V shaders/TexturedQuad.frag -o build/Shaders/TexturedQuad.frag.spv
glslangValidator -V shaders/FillTexture.comp -o build/Shaders/FillTexture.comp.spv
glslangValidator -V shaders/triangle.vert -o build/Shaders/triangle.vert.spv
glslangValidator -V shaders/triangle.frag -o build/Shaders/triangle.frag.spv
glslangValidator -V shaders/colormap.comp -o build/Shaders/colormap.comp.spv
glslangValidator -V shaders/findmax.comp -o build/Shaders/max.comp.spv
glslangValidator -V shaders/findmin.comp -o build/Shaders/min.comp.spv
glslangValidator -V shaders/firstmax.comp -o build/Shaders/firstmax.comp.spv
glslangValidator -V shaders/firstmin.comp -o build/Shaders/firstmin.comp.spv
glslangValidator -V shaders/transferandsquare.comp -o build/Shaders/transferandsquare.spv
