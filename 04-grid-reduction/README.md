# Grid reduction
This requires reduction across thread blocks, there are generally two approaches:

* Launch a second kernel (kernel launch is a natural synchronization point by default)
* Use just one kernel, but with atomic operations (serialized) such as `atomicAdd`.

In this tutorial we use `atomicAdd` to achieve reduction across thread blocks.