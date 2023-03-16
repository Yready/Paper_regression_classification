"""
-*- coding: utf-8 -*-
utils-
Author-木易杨
Date-2022/10/18
"""
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


# https://www.kaggle.com/code/gpiosenka/eye-disease-f1-score-92-6
def reg_plot(tr_data, save_path, start_epoch=0):
    # Plot the training and validation data
    mae = tr_data.history['mae']
    tloss = tr_data.history['loss']
    val_mae = tr_data.history['val_mae']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(mae) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)
    index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmin(val_mae)
    acc_highest = val_mae[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].scatter(Epochs, tloss, s=100, c='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs', fontsize=18)
    axes[0].set_ylabel('Loss', fontsize=18)
    axes[0].legend()

    axes[1].plot(Epochs, mae, 'r', label='Training MAE')
    axes[1].scatter(Epochs, mae, s=100, c='red')
    axes[1].plot(Epochs, val_mae, 'g', label='Validation MAE')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation MAE')
    axes[1].set_xlabel('Epochs', fontsize=18)
    axes[1].set_ylabel('MAE', fontsize=18)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()


def cls_plot(tr_data, save_path, start_epoch=0):
    # Plot the training and validation data
    tacc = tr_data.history['acc']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_acc']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)
    index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].scatter(Epochs, tloss, s=100, c='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs', fontsize=18)
    axes[0].set_ylabel('Loss', fontsize=18)
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].scatter(Epochs, tacc, s=100, c='red')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs', fontsize=18)
    axes[1].set_ylabel('Accuracy', fontsize=18)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()


# 自制数据集函数
def generateds(path, txt):  # 通过函数导入数据路径和
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开保存到value中，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼接图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_GRAY2RGB)  # 将灰度图像转为彩色图像, 如果图像为彩色图像，直接删除此行
        img = img / 255.  # 数据归一化（实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y.append(value[1])  # 标签贴到列表y_

    x = np.array(x)  # x变为np.array格式
    y = np.array(y)  # y变为np.array格式
    y = y.astype(np.int64)  # y变为64位整型
    return x, y  # 返回输入特征x，返回标签y_


class step_Loss(tf.keras.callbacks.Callback):
    def __init__(self):
        self.metrics = None
        self.loss = None

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.metrics = []
        self.loss = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.metrics.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))


class show_lr(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def show(self, logs=None):
        current_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print('current lr:{:0.7f}'.format(current_lr))


