## Build instructions
Required dependencies: Vulkan, opencv, glslang, VulkanMemoryAllocator
```
cmake -B build
./compileEverything.sh
```
You may need to make `compileEverything.sh` executable: `chmod +x compileEverything.sh`.
## To run
Run `x` steps
```
./VulkanCompute -n x
```
Run one step at a time (press `n` with the display window open to step):
```
./VulkanCompute -d
```

