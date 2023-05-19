# Reduction using warp shuffle
Previously we used shared memory to communicate between threads in a block. Wouldn't it be good if there's a hardware mechanism that lets threads directly communicate with each other (in registers) without going through shared memory? There is actually a family of such warp level synchronization intrinsics.

# Further reading
* https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
* https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
* https://vimeo.com/419029739

