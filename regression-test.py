"""
-*- coding: utf-8 -*-
测试-
Author-木易杨
Date-2022/10/16
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
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

model_name = 'D:/Users/OPTIC/论文/英文/偏心0.1/regression-ecc0.1.h5'
plt_scatter = "D:/Users/OPTIC/论文/英文/无偏心-间隙/750_test.png"
# 根据输入特征和标签，自制数据集
# 导入数据集，添加数据集特征和标签路径，以及保存的路径
test_path = 'D:/Users/OPTIC/论文/英文/偏心0.1/拟合-测试集/'      # 测试集输入特征路径
test_txt = 'D:/Users/OPTIC/论文/英文/偏心0.1/拟合-测试集/regression_test.txt'    # 测试集标签文件

x_test, y_test = generateds(test_path, test_txt)

# X_train = [cv2.cvtColor(cv2.resize(i, (256, 256)), cv2.COLOR_GRAY2RGB) for i in x_train]
# X_test = [cv2.cvtColor(cv2.resize(i, (256, 256)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_test = np.asarray(x_test).astype('float32')

print(x_test.shape)

y_test_pd = pd.DataFrame(y_test)

y_test_pd.columns = ['label']

Periodic_range = np.linspace(0, 0.375, 61)
y_test_pd['true_label'] = Periodic_range[y_test_pd['label']]


model = load_model(model_name)

predicted = model.predict(x_test)
y_pred_pd = pd.DataFrame({'label': list(y_test_pd['label']), 'pred_label': list(predicted.reshape(len(predicted),))})  # todo
y_test_pd['pred_label'] = pd.DataFrame(y_pred_pd['pred_label'])

print("RMSE:", mean_squared_error(y_test_pd['true_label'], y_test_pd['pred_label'], squared=False))

plt.scatter(y_test_pd['true_label'], y_test_pd['pred_label'])
x = np.linspace(0, 0.750/2, 100)
y = x
plt.plot(x, y, color='red', linewidth=1.0, linestyle='--', label='line')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.legend(["y = x", "预测值"])
plt.title("预测值与真实值的偏离程度")
plt.xlabel('真实值')
plt.ylabel('预测值')
# plt.savefig(plt_scatter, dpi=200, bbox_inches='tight', transparent=False)
plt.show()
