"""
-*- coding: utf-8 -*-
论文predict-
Author-木易杨
Date-2022/10/20
"""
import os
from time import *
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import generateds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

begin_time = time()
csv_path = 'D:/Users/OPTIC/论文/英文/无偏心-间隙/论文预测-柱状图.csv'                      # todo
surplus_path = 'D:/Users/OPTIC/论文/英文/无偏心-间隙/serial.csv'                         # todo
print("[INFO] load model.h5...")
model_only = load_model("D:/Users/OPTIC/论文/英文/无偏心-间隙/regression.h5")  # todo
model_two = load_model("D:/Users/OPTIC/论文/英文/无偏心-间隙/cls.h5")  # todo

print("[INFO] load dataset...")
only_path = 'D:/Users/OPTIC/论文/英文/无偏心-间隙/单张图片/'    # 训练集输入特征路径    # todo
only_txt = 'D:/Users/OPTIC/论文/英文/无偏心-间隙/单张图片/label.txt'                # todo
only_x, _ = generateds(only_path, only_txt)
only_x = np.asarray(only_x).astype('float32')

# 数据生成
test_datagen = ImageDataGenerator(rescale=1. / 255)
two_x = test_datagen.flow_from_directory(
    'D:/Users/OPTIC/论文/英文/无偏心-间隙/拼接图片',                               # todo
    target_size=(64, 128),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)

true_labels = two_x.class_indices  # 查看类别的label

test = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/Users/OPTIC/论文/英文/无偏心-间隙/分类",                                  # todo
    label_mode='categorical',
    image_size=(64, 128),
    shuffle=False,
    batch_size=1)
P_range = np.array(test.class_names)

print("[INFO] data predict...")
pred_only = model_only.predict(only_x, verbose=1)
pred_two = model_two.predict(two_x, verbose=1)

end_time = time()
run_time = end_time - begin_time
print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间

pred_two = pd.DataFrame(np.argmax(pred_two, axis=1))
pred_two.columns = ['label']

pred_two['pred_labels'] = P_range[pred_two['label']]
pred_two['pred_labels'] = np.array(list(map(int, pred_two['pred_labels'])))  # 将字符串转换为int类型

pred_labels = pred_two['pred_labels']*0.375 + pred_only.reshape(len(pred_only),)

true_labels = np.array(list(map(float, true_labels)))  # 将字符串转换为float类型
cha = pred_labels - true_labels
surplus = np.array([i for i in cha if abs(i) < 0.375])
print('低于100nm的个数为：', np.size(surplus))
rmse = np.sqrt(sum(surplus * surplus)/len(surplus))

print('RMSE:', rmse)
a = []
for i in range(200):
    a.append(np.sum(abs(surplus) < 0.00375/2*(i+1))-np.sum(abs(surplus) <= 0.00375/2*i))

np.savetxt(csv_path, a, delimiter=',')
np.savetxt(surplus_path, abs(cha), delimiter=',')

