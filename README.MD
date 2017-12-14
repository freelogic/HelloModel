# README.MD

## 简述
1. 本项目期望build一个较小的简单model，并用少量图片训练，来看weight的变化。
2. 本项目用到了quiver可视化库（安装方法见下面作者录制的视频资料）。
3. [model及参数计算: layer_analyse.md](https://github.com/freelogic/HelloModel/blob/master/platform/keras/HelloModel/layer_analyse.md)。


## 资料
(说明：部分资料来源于网络，如果有LICENSE等问题，请联系我们)
1. keras source code：https://github.com/fchollet/keras/blob/master/keras/engine/topology.py
2. keras chinese userguide：http://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/
3. How to install keras-sequential-ascii:
    pip cmd: pip install git+git://github.com/stared/keras-sequential-ascii.git
    source codes: https://github.com/stared/keras-sequential-ascii
4. How to install quiver for CNN visualize， please see my simple demo from below：
   name：quiver_engine_install_demo-by-taichu.swf
   video：https://pan.baidu.com/s/1mi5S59e  （free and valid forever depends on pan.baidu）

## TODO
1. TODO：Try to use layer class in keras, which has function "get_output_at(node_index),get_output_shape_at(node_index),.."
2. TODO：计划增强开源库keras_sequential_ascii的输出，自动计算神经元个数和参数个数及计算过程；
3. TODO: 增加HelloModel项目的矩阵内容输出，用简单的小量数据来观察计算过程及矩阵数据变化；
4. TODO：利用本项目来测试自定义层layer的前向和后向效果（比如开发一个dropout层，带有阶梯聚类和阈值清空的功能）；
    
