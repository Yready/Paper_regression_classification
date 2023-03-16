"""
-*- coding: utf-8 -*-
config-
Author-木易杨
Date-2022/10/25
"""
image_size = (256, 256)

# 波长750
train_data_750 = './750_regression_dataset/train'    # 训练集输入特征路径\
val_data_750 = './750_regression_dataset/validation'      # 测试集输入特征路径

train_txt_750 = './750_regression_dataset/train/regression_train.txt'  # 训练集标签txt文件
val_txt_750 = './750_regression_dataset/validation/regression_validation.txt'    # 测试集标签文件
model_name_750 = '750_regression_test__.h5'
loss_curves_750 = './Paper_regression/750_regression.png'
plt_scatter_750 = "./Paper_regression/750_test_xx.png"
path_checkpoint_750 = "./Paper_regression/750_weights.{epoch:02d}-{val_loss:.2f}.h5"


# 波长737
train_data_737 = './737_regression_dataset/train/'    # 训练集输入特征路径\
val_data_737 = './737_regression_dataset/validation/'      # 测试集输入特征路径

train_txt_737 = './737_regression_dataset/train/regression_train.txt'  # 训练集标签txt文件
val_txt_737 = './737_regression_dataset/validation/regression_validation.txt'    # 测试集标签文件

model_name_737 = '737_regression_test__.h5'
loss_curves_737 = './Paper_regression/737_regression.png'
plt_scatter_737 = "./Paper_regression/737_test_xx.png"
path_checkpoint_737 = "./Paper_regression/737_weights.{epoch:02d}-{val_loss:.2f}.h5"

batch_size = 32


# train
base_lr = 0.0001
total_epochs = 100
CONTINUE = False
start_epoch = 0

# evaluate
log_epoch = 2
log_loss = 100