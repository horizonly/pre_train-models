# -*- coding: UTF-8 -*-
_author_ = 'horizon'
_date_ = '2019/12/19'
#requirements: tensorflow-gpu 1.11.0  keras-gpu 2.2.4  pillow 6.2.1
#---------------keras实现VGG16加载预训练权重提取图片特征--------------------#
import os
from time import *
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

conv_base = VGG19(weights = 'imagenet', #模型初始化权重检查点
                  include_top=False,    #最后是否包含全连接分类器
                  input_shape=(224, 224, 3))  #形状可选 不设置则可处理任意大小图片

#print(conv_base.summary())  #输出VGG16网络结构

base_dir = '/home/hesongze/PycharmProjects/pre_network'
picture_dir = os.path.join(base_dir, 'data')

datagen = ImageDataGenerator(rescale = None)  #定义一个处理图像的方案 rescale缩放因子
batch_size = 5

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))
    generator = datagen.flow_from_directory(directory, #从该路径下读取到类别文件 自动生成label
                                            target_size = (224, 224),
                                            batch_size = batch_size,
                                            class_mode = 'binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)  #提取特征的关键部分
        features[i * batch_size : (i + 1) * batch_size] = features_batch #一个个batch往features里存放
        i += 1
        if i *batch_size >= sample_count: #大于样本总数则跳出循环
            break   #生成器在循环中不断生成数据，必须在读完所有图像后终止循环
    return features

#begin = time()
train_features = extract_features(picture_dir, 4000)
#end = time()
#print(end - begin)
train_features = np.reshape(train_features, (4000, 7*7*512))
np.save('feature.npy', train_features)  #数据存储
