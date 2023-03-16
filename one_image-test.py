"""
-*- coding: utf-8 -*-
单张预测-
Author-木易杨
Date-2022/10/25
"""
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
# ignoring warnings
import warnings

warnings.simplefilter("ignore")

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
print('Tensorflow Version: {}'.format(tf.__version__))
print('GPU', tf.config.list_physical_devices('GPU'))

test_ds_737 = tf.keras.preprocessing.image_dataset_from_directory(
        "D:/Download/数据集/737分类/train",
        label_mode='categorical',
        image_size=(256, 256),
        batch_size=1)

class_names_737 = test_ds_737.class_names

test_ds_750 = tf.keras.preprocessing.image_dataset_from_directory(
        "D:/Download/数据集/750分类/train",
        label_mode='categorical',
        image_size=(256, 256),
        batch_size=1)

class_names_750 = test_ds_750.class_names


def gen(img_path, model_path):
    img = Image.open(img_path)
    img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
    img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_GRAY2RGB)
    img = img / 255.  # 数据归一化（实现预处理）
    x = [img]
    x = np.array(x)  # x变为np.array格式
    x = np.asarray(x).astype('float32')
    model = load_model(model_path)
    pred = model.predict(x)
    return pred


model_750 = './分类737.h5'
model_737 = './750分类.h5'

img_737 = 'D:/两个网络/测试/0.737_0.006.bmp'
img_750 = 'D:/两个网络/测试/0.75_0.006.bmp'
pred_737 = class_names_737[np.argmax(gen(img_737, model_737))]
pred_750 = class_names_750[np.argmax(gen(img_750, model_750))]
print(pred_737, pred_750)
# WL737 = np.linspace(0, 56, 57)*0.737/2
# WL750 = np.linspace(0, 56, 57)*0.750/2
# pred_750 = gen(img_750, model_750)
# pred_737 = gen(img_737, model_737)
# print(pred_737, pred_750)
#
# cha = abs((WL750+pred_750)-(WL737+pred_737))
# print(cha.min())
# pred = (WL750[np.argmin(cha)]+pred_750+WL737[np.argmin(cha)]+pred_737)/2
# print(pred)