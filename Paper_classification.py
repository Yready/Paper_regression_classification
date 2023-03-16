"""
-*- coding: utf-8 -*-
单波长分类-
Author-木易杨
Date-2022/10/26
"""
import os
import config
from utils import *
from tensorflow.keras import applications, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

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

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_ds = train_datagen.flow_from_directory(
    "D:/Download/数据集/N个分类-WT6/train",
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

val_ds = val_datagen.flow_from_directory(
    "D:/Download/数据集/N个分类-WT6/validation",
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')


base_model = mobilenet.MobileNet(input_shape=(256, 256, 3),
                                 include_top=False,
                                 weights='imagenet')
model = tf.keras.models.Sequential([
    base_model,
    # 对主干模型的输出进行全局平均池化linea
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(56, activation='softmax')
])
model.summary()
# # 保持VGG16的前15层权值不变，即在训练过程中不训练
# for layer in model.layers[:15]:
#     layer.trainable = False
# 模型训练的优化器为adam优化器，模型的损失函数为交叉熵损失函数
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='acc')

his = step_Loss()

history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=his)
np.savetxt('./Paper_regression/cls_loss.csv', his.loss, delimiter=',')
np.savetxt('./Paper_regression/cls_acc.csv', his.metrics, delimiter=',')
model.save('cls_MobileNetV1.h5')
cls_curves = './Paper_regression/拼接.png'
cls_plot(history, cls_curves)
