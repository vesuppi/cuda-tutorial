# 简介
一个简明的GPU编程入门教程，主要使用三个例子：向量加法、求和以及softmax。每个例子除了CUDA版本以外，也同时有一个[triton](https://triton-lang.org/main/index.html)的版本。本教程一共6节，内容依次涵盖

* 第一节：向量加法
  * 感受GPU编程的思维
* 第二节：矩阵按行求和
  * 让相邻的线程访问相邻的内存地址来避免常见的性能问题
  * 使用shared memory来进行线程之间的同步
* 第三节：矩阵按行求和（继续）
  * 使用warp shuffle来进行warp线程之间的同步
* 第四节：矩阵全局求和
  * 使用atomic操作来进行线程块之间的同步
* 第五节：softmax
  * 一个既有element-wise也有reduction操作的例子
  * 使用shared memory作为缓存
* 第六节：矩阵乘法
  * 内存访问优化的经典例子
  * 使用triton来简化分块的矩阵乘法编程

# 适用人群
有一定C++编程和Python编程基础，希望入门GPU并行编程。

# 运行
先安装好环境：
```bash
pip install -e .
```

然后运行一个章节的例子：
```bash
python 01-vector-add/main.py
```

库依赖清单（运行`pip install -e .`时会自动安装）：

- `torch`用来管理数据的创建和性能测试
- `cupy`使用简化CUDA kernel的编译和调用
- `triton`用来编写triton kernel

# 为什么要使用GPU？为什么CPU不够高效？
<!--
在过去的几十年中，CPU的性能一直在持续地提高，那我们为什么还需要GPU呢？等着过几年CPU的性能继续提高不就行了吗？

我们首先来看看CPU的性能是通过哪几个方面提升的，主要是下面三个方面：

* 更高的时钟频率（clock frequency）
* 更多的指令级并行（instruction level parallelism，或ILP）
* 更多的核心数

我们知道，晶体管的制造工艺不断在进步，使得每隔一两年我们就能把晶体管做得更小，一直到现在。那晶体管的大小和CPU的时钟频率是什么关系呢？我们来看下面这张图（来源：https://github.com/karlrupp/microprocessor-trend-data）：

![processor-trend](https://www.karlrupp.net/wp-content/uploads/2018/02/42-years-processor-trend.png)

这里我们可以发现，在大约2006年以前，随着晶体管做得越来越小，晶体管的数量和CPU的频率都在指数增长。这里面有两个“定律”，一个是[摩尔定律](https://en.wikipedia.org/wiki/Moore%27s_law)，它是说晶体管的数量大约每2年翻一倍（通过把晶体管做得越来越小），同时保持芯片价格不变。但摩尔定律并不直接指出晶体管数量和CPU性能的关系。第二个是[Dennard scaling](https://en.wikipedia.org/wiki/Dennard_scaling)，它指出，当晶体管的大小变小的时候，在不增加能耗的情况下，我们能够在单位面积里堆更多的晶体管，而且同时还能提升时钟频率。这个规律导致了从1970年代到2006年CPU频率（和性能）的指数提升。

但是大约在2006年前后，Dennard scaling结束了，我们不再能够持续降低晶体管的电压（现在电压基本保持不变），同时也不能能够在不增加能耗的情况下提升CPU的频率，尽管晶体管的数量还在持续增长。而摩尔定律时至今日，依然继续，尽管速度有放缓，但是CPU的晶体管数量依然在持续增加。
-->

CPU的设计使得它对于大量数值计算并不高效，CPU的首要目标是提升串行程序的性能，于是它使用了非常复杂的控制逻辑来提升ILP，譬如分支预测、乱序执行等等，这些东西占据了大量的晶体管。除此之外为了hide memory latency，CPU也会使用很大的缓存。最后的结果就是“CPU optimize for latency”。

GPU的设计采用了另一种思路。很多的应用其实都是对大量的数据进行同样的操作，对于这样的应用，我们在意的其实是算完所有数据的总时间，而不在意操作其中一个数据的快慢。于是GPU把在CPU里面用来堆控制逻辑（和缓存）的那些晶体管都堆成了计算单元（ALU），当然GPU也有缓存，不过相对CPU要更小。这样的设计被称为“latency hiding”。

以下图片来自于CUDA C编程指南（https://docs.nvidia.com/cuda/cuda-c-programming-guide/）：

![GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)

# 教程简介
本教程一共6节，内容依次涵盖

* 第一节：向量加法
  * 感受GPU编程的思维
* 第二节：矩阵按行求和
  * 让相邻的线程访问相邻的内存地址来避免常见的性能问题
  * 使用shared memory来进行线程之间的同步
* 第三节：矩阵按行求和（继续）
  * 使用warp shuffle来进行warp线程之间的同步
* 第四节：矩阵全局求和
  * 使用atomic操作来进行线程块之间的同步
* 第五节：softmax
  * 一个既有element-wise也有reduction操作的例子
  * 使用shared memory作为缓存
* 第六节：矩阵乘法
  * 使用triton来简化分块的矩阵乘法编程

希望用这三个例子来阐明GPU编程的几个要点：

* 对问题的并行拆解
* 相邻的线程访问相邻的内存地址
* 使用shared memory进行通信和缓存

另外传统的CUDA编程教程基于C++，而本教程的driver程序使用Python编写，方便数据创建以及性能测试等等，只有CUDA kernel本身使用CUDA C++，这样旨在简化周边代码而更多关注GPU编程的本质（并行算法设计）。每一个例子除了CUDA的版本我们也编写了一个[triton](https://triton-lang.org/main/index.html)的版本，这使得读者可以看到两者的区别，以及各自的pros and cons。譬如CUDA的适用性更广，对每个线程有更精确的控制，而triton则对于特定类型的程序写起来更快，简化了编程。这样的对比也帮助读者思考，我们到底使用什么样的方式对GPU进行编程？到底如何兼顾性能、广泛性和生产力？我们希望我们的GPU编程语言能够表达许多场景的程序，也希望它足以发挥出硬件的性能，同时我们也不希望编程过于繁琐。

# 扩展阅读
本教材的目的仅仅是让读者对GPU计算有一个初步的了解，我们使用简单的矩阵计算为例，借此希望能让读者感受GPU计算的魅力，但GPU不仅仅是可以用来加速深度学习，它还有各种各样其他的应用场景，事实上我们也希望读者能够去思考如何为自己领域的某个算法设计一个GPU并行的版本。

对于需要了解更深入的CUDA编程特性，以及如何用GPU来解决更多的问题（譬如排序），我们我们推荐下面的阅读和教程：
* [CUDA Training Series](https://www.olcf.ornl.gov/cuda-training-series/)
  * 我看过最好的CUDA入门教程，录像在每一个章节的页面里
  * 前面的三章比较基本，后面是更加深入的内容
* [Intro to Parallel Programming](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2&ab_channel=Udacity)
  * 主要讨论GPU算法，包括排序等
  * 除了教程本身，里面的人物也很棒
* [ETH:Hands-on Acceleration on Heterogeneous Computing Systems](https://safari.ethz.ch/projects_and_seminars/spring2022/doku.php?id=heterogeneous_systems)
  * 这个更加学术一点，譬如它会告诉你SIMT的本质是用户编写SPMD、而机器动态进行SIMD，以及GPU使用fine-grain multi-threading来hide latency等

以下来自NVIDIA官方的书比较系统和庞大，可以作为参考书：

* https://developer.nvidia.com/cuda-example
* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
* https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf
* https://docs.nvidia.com/cuda/parallel-thread-execution/index.html