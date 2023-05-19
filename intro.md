# 为什么要使用GPU？

在过去的几十年中，CPU的性能一直在持续地提高，那我们为什么还需要GPU呢？等着过几年CPU的性能继续提高不就行了吗？

我们首先来看看CPU的性能是通过哪几个方面提升的，主要是下面三个方面：

* 更高的时钟频率（clock frequency）
* 更多的指令级并行（instruction level parallelism，或ILP）
* 更多的核心数

我们知道，晶体管的制造工艺不断在进步，使得每隔一两年我们就能把晶体管做得更小，一直到现在。那晶体管的大小和CPU的时钟频率是什么关系呢？我们来看下面这张图（来源：https://github.com/karlrupp/microprocessor-trend-data）：

![processor-trend](https://www.karlrupp.net/wp-content/uploads/2018/02/42-years-processor-trend.png)

这里我们可以发现，在大约2006年以前，随着晶体管做得越来越小，晶体管的数量和CPU的频率都在指数增长，这个过程被称为[Dennard scaling](https://en.wikipedia.org/wiki/Dennard_scaling)，值得一提的是，Robert Dennard也发明了DRAM（现年90岁）。

Dennard scaling指出了这样一个规律，当晶体管的dimension变成0.7倍，面积大约减小一半（0.7乘0.7），电压也变为0.7倍，这时候将CPU频率变为大约1.4倍，而晶体管的power density不变，即同样的面积现在堆了两倍的晶体管，但是总体的能耗是不变的。也就是说，在不增加单位面积的能耗的情况下，我们即可以堆之前两倍的晶体管数量，同时又能把时钟频率变成之前的1.4倍。这个规律导致了从1970年代到2006年CPU频率（和性能）的指数提升。

然而Dennard scaling也有结束的一天，这里的问题在于，电压降到一定程度就无法再继续下降了

* 晶体管的面积变成了大约0.5倍（0.7乘0.7），即面积减小一半
* 电容（C）变为0.7倍
* 电压（V）也变为0.7倍


![Dennard](https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Robert_Dennard.jpg/330px-Robert_Dennard.jpg)


但是大约在2006年前后，这个趋势停止了，CPU的频率不再继续增长，尽管晶体管的数量还在持续增长。这个原因并不是因为摩尔定律停止了，事实上摩尔定律说的只是晶体管的数量，而不是CPU“单核性能”或者“频率”。

# 三大要点
* 对问题的并行拆解
* 相邻的线程访问相邻的内存地址
* 使用shared memory进行通信和缓存