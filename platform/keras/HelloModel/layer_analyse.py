'''
added by taihcu
1.Let ONLY ONE picture through the model,do forward computing and back-propagation(BP) and SEE how matrix updated!
2.Try to use layer class in keras, which has function "get_output_at(node_index),get_output_shape_at(node_index),.."
3.keras source code：https://github.com/fchollet/keras/blob/master/keras/engine/topology.py
4.keras chinese userguide：http://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/
5.install keras-sequential-ascii:
    pip cmd: pip install git+git://github.com/stared/keras-sequential-ascii.git
    source codes: https://github.com/stared/keras-sequential-ascii


'''
from __future__ import print_function

import keras
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
# from keras import optimizers
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Activation, MaxPool2D
from keras.callbacks import LearningRateScheduler, TensorBoard, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
# from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras_sequential_ascii import sequential_model_to_ascii_printout
import argparse,os,glob,math
from tool.utils import get_nb_files
from quiver_engine import server

config = tf.ConfigProto()
config = tf.ConfigProto(log_device_placement=False, device_count={'gpu':0})  # 设定与否都默认用GPU:0
# config = tf.ConfigProto(log_device_placement=True,device_count={'cpu':0}) #设定与否都取法启用CPU:0
# config.gpu_options.allow_growth=True
# config.log_device_placement=True #可以指示出变量对设备的安排（GPU/CPU等）；
# config.device_count=({'gpu':0}) #初始化config之后再用此句设定会引起重复定义device:0设备的报错。
KTF.set_session(tf.Session(config=config))

np.random.seed(0)  # 设定随机种子，据说能让程序重现结果。
print("Initialized!")

# 定义变量
batch_size = 2
epochs = 2
nb_classes = 2
IM_WIDTH, IM_HEIGHT = 16, 16
nb_filters = [2, 3, 1, 1, 1]
pool_size = (2, 2)
kernel_size = (3, 3)
lr = 0.0001
log_filepath = './layer_analyse'
ag_ratio = 1.0
weight_decay  = 0.0001

class CustomMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            if k.endswith('l1_conv2d'):
               print(logs[k])

def parselayer(layer, node_index=0):
    layer.get_weights()  # 返回该层的权重
    #layer.set_weights(weights)  # 将权重加载到该层
    config = layer.get_config()  # 保存该层的配置
    #layer = layer_from_config(config)  # 加载一个配置到该层
    # 该层有一个节点时，获得输入张量、输出张量、及各自的形状：
    #layer.input
    print(layer.output)
    #layer.input_shape
    print(layer.output_shape)
    # 该层有多个节点时（node_index为节点序号）：
    #layer.get_input_at(node_index)
    print(layer.get_output_at(node_index))
    #layer.get_input_shape_at(node_index)
    print(layer.get_output_shape_at(node_index))


def build_model(input_shape):
    model = Sequential()
    l1_conv2d = Conv2D(nb_filters[0], kernel_size, padding='valid', input_shape=input_shape,name="l1_conv2d",)
    l2_act_relu = Activation('relu',name="l2_act_relu")
    l3_conv2d = Conv2D(nb_filters[1], kernel_size,strides=[2,2] , name="l3_conv2d", padding='valid')
    l4_act_relu = Activation('relu',name="l4_act_relu")
    l5_mpool2d = MaxPool2D(pool_size=pool_size,name="l5_mpool2d")
    l6_dropout = Dropout(0.1,name="l6_dropout")
    l7_flatten = Flatten(name="l7_flatten")
    l8_dense = Dense(5,name="l8_dense")
    l9_act_relu = Activation('relu',name="l9_act_relu")
    l10_dense = Dense(nb_classes,name="l10_dense")
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
    '''
    #model.add(Conv2D(nb_filters[0], kernel_size, padding='same',kernel_regularizer=l2(weight_decay), input_shape=input_shape))
    l1_conv2d = Conv2D(nb_filters[0], kernel_size, padding='same', input_shape=input_shape,name="l1_conv2d",)
    model.add(l1_conv2d)
    #parselayer(l1_conv2d) 放这里没用！
    l2_act_relu = Activation('relu',name="l2_act_relu")
    model.add(l2_act_relu)
    #model.add(LeakyReLU(alpha=0.5))
    #model.add(BatchNormalization())
    #model.add(Conv2D(nb_filters[1], kernel_size,kernel_regularizer=l2(weight_decay)))
    l3_conv2d = Conv2D(nb_filters[1], kernel_size,strides=[3,3] , name="l3_conv2d", padding='same')
    model.add(l3_conv2d)
    l4_act_relu = Activation('relu',name="l4_act_relu")
    model.add(l4_act_relu)
    #model.add(LeakyReLU(alpha=0.5))
    #model.add(BatchNormalization())
    l5_mpool2d = MaxPool2D(pool_size=pool_size,name="l5_mpool2d")
    model.add(l5_mpool2d)
    l6_dropout = Dropout(0.1,name="l6_dropout")
    model.add(l6_dropout)

    #model.add(Conv2D(nb_filters[2], kernel_size, padding='same',kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(nb_filters[2], kernel_size, padding='same'))
    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha=0.5))
    #model.add(BatchNormalization())
    #model.add(Conv2D(nb_filters[3], kernel_size,kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(nb_filters[3], kernel_size))
    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha=0.5))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    #model.add(Conv2D(nb_filters[4], kernel_size, padding='same',kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(nb_filters[4], kernel_size, padding='same'))
    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha=0.5))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(Dropout(0.01))
    
    
    l7_flatten = Flatten(name="l7_flatten")
    model.add(l7_flatten)
    #model.add(Dense(512, kernel_regularizer=l2(weight_decay)))
    l8_dense = Dense(5,name="l8_dense")
    model.add(l8_dense)
    l9_act_relu = Activation('relu',name="l9_act_relu")
    model.add(l9_act_relu)
    #model.add(LeakyReLU(alpha=0.5))
    #model.add(Dropout(0.5))
    #model.add(Dense(nb_classes, kernel_regularizer=l2(weight_decay)))
    l10_dense = Dense(nb_classes,name="l10_dense")
    model.add(l10_dense)
    l11_act_relu = Activation('softmax',name="l11_act_relu")
    model.add(l11_act_relu)

    '''

    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model

def lr_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lr

def imshow(img):  # input numpy.ndarray
    plt.subplot(1, 1, 1)
    plt.imshow(img)
    # plt.axis("off")
    plt.show()

def train(args):
    """load data, natural classify DEPEND BY DIR split!!!"""
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    epochs = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    # build network
    #model = build_model((IM_WIDTH, IM_HEIGHT,1))
    model = build_model((IM_WIDTH, IM_HEIGHT, 3))
    print(model.summary())  # keras方便打印model的语句

    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(lr_decay) #设定了学习率开始太大，导致根据训练不上去！
    cnn_normal_model = ModelCheckpoint("layer_analyse.h5", monitor='val_loss', verbose=0,
                                       save_best_only=True)
    #cbks = [cnn_normal_model, tb_cb, change_lr]
    cbks = [cnn_normal_model, tb_cb]

    '''
    # using real-time data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)
    '''
    # refer：http://blog.csdn.net/u012969412/article/details/76796020 (keras的ImageDataGenerator详细参数说明）
    # refer：http://www.sohu.com/a/198572539_697750 #极其详细的图示每个参数；
    # refer：http://keras-cn.readthedocs.io/en/latest/preprocessing/image/ #keras官网解释；
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        rescale=1 / 255,
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # 角度旋转很有必要，cifar10的图片角度都不同，提升model泛化能力；
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.20,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.20,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images # cifar10的图片如cat左右朝向都有，所以引入随机左右翻转；
        vertical_flip=False,  # randomly flip images # cifar10的图片一般不会上下颠倒，比如cat倒过来；
        fill_mode='constant', cval=0.,
        zoom_range=[0.8, 1.0])  # randomly zoom size,超过1.0倍就是图像局部被学习，不合理，只能缩小，不能大于原尺寸；
    # 不用设定rescale，因为之前已经做过了dp预处理（X=(X-MEAN)/STD）

    test_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        rescale=1 / 255,
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # 角度旋转很有必要，cifar10的图片角度都不同，提升model泛化能力；
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images # cifar10的图片如cat左右朝向都有，所以引入随机左右翻转；
        vertical_flip=False,  # randomly flip images # cifar10的图片一般不会上下颠倒，比如cat倒过来；
        fill_mode='constant', cval=0.,
        zoom_range=[0.9, 1.0])  # randomly zoom size,超过1.0倍就是图像局部被学习，不合理，只能缩小，不能大于原尺寸；
    # 不用设定rescale，因为之前已经做过了dp预处理（X=(X-MEAN)/STD）

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        #batch_size=batch_size,color_mode='grayscale'
        batch_size = batch_size, color_mode = 'rgb'
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        #batch_size=batch_size,color_mode='grayscale'
        batch_size=batch_size, color_mode='rgb'
    )

    #如下语句安装ascii版本的model格式打印，也别有风味；
    #pip install git+git://github.com/stared/keras-sequential-ascii.git
    #https://github.com/stared/keras-sequential-ascii
    sequential_model_to_ascii_printout(model)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train) #见官网，仅当ZCA,或featurewise_XXX参数要求被提供时需要fit函数，其他情况可以不用。
    # start traing, Fit the model on the batches generated by datagen.flow().
    #model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    #                        steps_per_epoch=iterations,
    #                        epochs=epochs,
    #                        callbacks=cbks,
    #                        verbose=1,
    #                        validation_data=(x_test, y_test))
    history = model.fit_generator(train_generator,
        steps_per_epoch=nb_train_samples/batch_size*ag_ratio,
        epochs=epochs,
        callbacks=cbks,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=nb_val_samples/batch_size,
        class_weight='auto')

    # 模型评分
    #score = model.evaluate(x_test, y_test, verbose=0)
    # 输出结果
    #print('Test score:', score[0])
    #print("Accuracy: %.2f%%" % (score[1] * 100))
    #print("Compiled!")

    # save model
    model.save('layer_analyse.h5')
    return model

def visualize_model(model):
    # 设定quiver_engine可视化for keras/CNN
    # 如果开启这个launch函数，怎会自动打开web page，将预先放入input_folder的图片动态计算出来，其机理是动态计算几次epoch？还是利用
    # 当前模型的.h5文件直接计算，这个不得而知！！！且函数launch会挂起程序，不再让GPU计算当前模型的200个epoch。所以代码如何增加quiver
    # 是要仔细考虑一下的！quiver的源码要看看！TODO
    # cifar10_classname = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    server.launch(
        model=model,  # a Keras Model
        # classes = cifar10_classname,
        # list of output classes from the model to present (if not specified 1000 ImageNet classes will be used)
        # top = 5,  # number of top predictions to show in the gui (default 5)

        # where to store temporary files generatedby quiver (e.g. image files of layers)
        temp_folder='./quiver/tmp',

        # a folder where input images are stored
        input_folder='./quiver/input_imgs',

        # the localhost port the dashboard is to be served on
        port=5000,

        # custom data mean
        # mean = [123.568, 124.89, 111.56]
        # mean = mean, #直接用cifar10的已经计算过的mean
        # custom data standard deviation
        # std = [52.85, 48.65, 51.56]
        # std = std  #直接用cifar10的已经计算过的std
    )

# run command example：
# cnn.py --train_dir=./train --val_dir=./val
# 数据集合是taichu自制为了测试GAN的猫狗小数据集合，详见《实践手册》，数据读入是用keras自动按dir目录分类的方式，很方便。
# 而不是类似mnist和cifar10的数据集合的情况。

if __name__ == '__main__':
        a = argparse.ArgumentParser()
        a.add_argument("--train_dir", default="train")
        a.add_argument("--val_dir", default="val")
        a.add_argument("--nb_epoch", default=epochs)
        a.add_argument("--batch_size", default=batch_size)
        args = a.parse_args()
        if args.train_dir is None or args.val_dir is None:
            a.print_help()
            sys.exit(1)

        if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
            print("directories do not exist")
            sys.exit(1)

        model = train(args)
        parselayer(model.get_layer("l1_conv2d"))

        #visualize_model(model)









