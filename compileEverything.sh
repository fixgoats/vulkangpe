#!/bin/bash
cmake --build build --parallel
glslangValidator -V shaders/rstep.comp -o build/rstep.spv
glslangValidator -V shaders/kstep.comp -o build/kstep.spv
glslangValidator -V shaders/finalstep.comp -o build/finalstep.spv
