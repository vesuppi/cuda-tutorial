一个简明的GPU编程入门教程，主要使用三个例子：向量加法、求和以及softmax。每个例子除了CUDA版本以外，也同时有一个triton的版本。
希望用这三个例子来阐明GPU编程的三个要点：

* 对问题的并行拆解
* 相邻的线程访问相邻的内存地址
* 使用shared memory进行通信和缓存

同时我们也会看到使用triton编程相对于cuda的简化。

# 为什么要使用GPU？

在过去的几十年中，CPU的性能一直在持续地提高，那我们为什么还需要GPU呢？等着过几年CPU的性能继续提高不就行了吗？

我们首先来看看CPU的性能是通过哪几个方面提升的，主要是下面三个方面：

* 更高的时钟频率（clock frequency）
* 更多的指令级并行（instruction level parallelism，或ILP）
* 更多的核心数

我们知道，晶体管的制造工艺不断在进步，使得每隔一两年我们就能把晶体管做得更小，一直到现在。那晶体管的大小和CPU的时钟频率是什么关系呢？我们来看下面这张图（来源：https://github.com/karlrupp/microprocessor-trend-data）：

![processor-trend](https://www.karlrupp.net/wp-content/uploads/2018/02/42-years-processor-trend.png)

这里我们可以发现，在大约2006年以前，随着晶体管做得越来越小，晶体管的数量和CPU的频率都在指数增长。这里面有两个“定律”，一个是[摩尔定律](https://en.wikipedia.org/wiki/Moore%27s_law)，它是说晶体管的数量大约每2年翻一倍（通过把晶体管做得越来越小），同时保持芯片价格不变。但摩尔定律并不直接指出晶体管数量和CPU性能的关系。第二个是[Dennard scaling](https://en.wikipedia.org/wiki/Dennard_scaling)，它指出，当晶体管的大小变小的时候，在不增加能耗的情况下，我们能够在单位面积里堆更多的晶体管，而且同时还能提升时钟频率。这个规律导致了从1970年代到2006年CPU频率（和性能）的指数提升。

但是大约在2006年前后，Dennard scaling结束了，我们不再能够持续降低晶体管的电压（现在电压基本保持不变），同时也不能能够在不增加能耗的情况下提升CPU的频率，尽管晶体管的数量还在持续增长。而摩尔定律时至今日，依然继续，尽管速度有放缓，但是CPU的晶体管数量依然在持续增加。

# 为什么CPU不够高效
除此之外，CPU的设计使得它对于大量数值计算并不高效。CPU的首要目标是提升串行程序的性能，于是它使用了非常复杂的控制逻辑来提升ILP，譬如分支预测、乱序执行等等，这些东西占据了大量的晶体管。除此之外为了hide memory latency，CPU也会使用很大的缓存。最后的结果就是“CPU optimize for latency”。

GPU的设计采用了另一种思路。很多的应用其实都是对大量的数据进行同样的操作，对于这样的应用，我们在意的其实是算完所有数据的总时间，而不在意操作其中一个数据的快慢。于是GPU把在CPU里面用来堆控制逻辑（和缓存）的那些晶体管都堆成了计算单元（ALU），当然GPU也有缓存，不过相对CPU要更小。这样的设计被称为“latency hiding”。

以下图片来自于CUDA C编程指南（https://docs.nvidia.com/cuda/cuda-c-programming-guide/）：

![GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)