# 模型层的参数分析

# 1简述
1. 好贴！卷积神经网络中的参数计算 ：https://www.cnblogs.com/hejunlin1992/p/7624807.html
2. 神经元个数和参数W及偏置bias不相同；神经元的输入可以是1到多个W及Bias;一般情况下神经元个数少于参数；
（对应到生物学，神经元比神经元的状态肯定少，假设一个神经元有多个状态，多个输入W/BIAS，等等）
3. CNN的卷积层conv2d，padding="same|valid"，strides步长，filter个数及大小，input的feature map个数，都会影响卷几层参数
4. TODO：计划增强开源库keras_sequential_ascii的输出，自动计算神经元个数和参数个数及计算过程；
5. TODO：增加HelloModel项目的矩阵内容输出，用简单的小量数据来观察计算过程及矩阵数据变化
## 2模型结构（基于keras_sequential_ascii库）
           OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####     16   16    3   input_shape=[16,16,3]，等于彩色图像的rgb三通道
              Conv2D    \|/  -------------------        56    21.1%  56=(3*3*3+1)*2=56=(卷集核3*3*输入维度 3+1bias）*输出的feature map个数2(也就是核数2)
                relu   #####     14   14    2   经过计算输出为2个feature map，padding='valid'所以不外拼0,16*16使用3*3的核只有14个输出位置；
              Conv2D    \|/  -------------------        57    21.5%  57=（3*3*2+1）*3=57
                relu   #####      6    6    3   padding='valid'且stride步长=2*2（只能从1,3,5,7,9，11开始感受野计算）,13开始就不够3*3的卷积核，因为是padding=valid不能补零，所以只有6个位置；
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      3    3    3   线性最大值池化，无参数
             Dropout    | || -------------------         0     0.0%
                       #####      3    3    3   普通随机概率dropout，无参数
             Flatten   ||||| -------------------         0     0.0%
                       #####          27        降维平展，从[3,3,3]展开未一维 27=3*3*3，无参数；
               Dense   XXXXX -------------------       140    52.8%
                relu   #####           5        全连接，27*5+5 = 140
               Dense   XXXXX -------------------        12     4.5%
             softmax   #####           2         全连接，5*2+2 = 12
        Total params: 265
        Trainable params: 265
        Non-trainable params: 0

## 3模型代码参考（可直接查看HelloModel/layer_analyse.py)
    #这里省略部分代码
    model = Sequential()
    l1_conv2d = Conv2D(2, [3,3], padding='valid', input_shape=[16,16,3],name="l1_conv2d",)
    l2_act_relu = Activation('relu',name="l2_act_relu")
    l3_conv2d = Conv2D(3, [3,3],strides=[2,2] , name="l3_conv2d", padding='valid')
    l4_act_relu = Activation('relu',name="l4_act_relu")
    l5_mpool2d = MaxPool2D(pool_size=[2,2],name="l5_mpool2d")
    l6_dropout = Dropout(0.1,name="l6_dropout")
    l7_flatten = Flatten(name="l7_flatten")
    l8_dense = Dense(5,name="l8_dense")
    l9_act_relu = Activation('relu',name="l9_act_relu")
    l10_dense = Dense(2,name="l10_dense")
    l11_act_relu = Activation('softmax',name="l11_act_relu")
    model.add(l1_conv2d)
    model.add(l2_act_relu)
    model.add(l3_conv2d)
    model.add(l4_act_relu)
    model.add(l5_mpool2d)
    model.add(l6_dropout)
    model.add(l7_flatten)
    model.add(l8_dense)
    model.add(l9_act_relu)
    model.add(l10_dense)
    model.add(l11_act_relu)
    #这里省略部分代码