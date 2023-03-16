"""
-*- coding: utf-8 -*-
分类测试-误差预测
Author-木易杨
Date-2023/2/17
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# 数据生成
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    'E:/服务器连接用/N个分类-偏心0.1/train',
    target_size=(64, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

model = load_model("D:/Users/OPTIC/论文/英文/偏心0.1/cls-ecc0.1.h5")

labels = test_generator.class_indices  # 查看类别的label

new_dict = {v: k for k, v in labels.items()}
# print(new_dict)
test_generator.reset()

pred = model.predict(test_generator, verbose=1)  # 然后直接用predict_geneorator 可以进行预测
predicted_class_indices = np.argmax(pred, axis=1)  # 输出每个图像的预测类别
true_label = test_generator.classes  # 测试集的真实类别
c = 0
for i in range(len(true_label)):
    b = float(new_dict[true_label[i]]) - float(new_dict[predicted_class_indices[i]])
    if b == 0:
        c += 1

print(c)
