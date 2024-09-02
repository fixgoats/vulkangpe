#!/bin/bash
cmake --build build --parallel
glslangValidator -V shaders/test.comp -o build/testing.spv
glslangValidator -V shaders/uhh.comp -o build/uhh.spv
