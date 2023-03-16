"""
-*- coding: utf-8 -*-
Paper_regression-https://blog.csdn.net/Chile_Wang/article/details/100556980?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166331715616800192257895%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166331715616800192257895&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-100556980-null-null.142^v47^pc_rank_34_default_2,201^v3^control_2&utm_term=%E5%9B%BE%E5%83%8F%E5%9B%9E%E5%BD%92&spm=1018.2226.3001.4187
Author-木易杨
Date-2022/5/11
"""
import os
import numpy as np
import pandas as pd
from time import *
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import reg_plot, generateds, step_Loss, show_lr
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import applications

# ignoring warnings
import warnings
warnings.simplefilter("ignore")

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# 获取所有 GPU 设备列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 GPU 显存占用为按需分配，增长式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 异常处理
        print(e)


# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     tf.config.experimental.set_memory_growth(gpu, True)
begin_time = time()
print('Tensorflow Version: {}'.format(tf.__version__))
print('GPU', tf.config.list_physical_devices('GPU'))

batch_size = 16
epochs = 2
# todo 数据位置更改
save_dir = os.path.join(os.getcwd(), 'Paper_regression')  # 使用os.getcwd()函数获得当前的路径
model_name = 'regression.h5'
loss_curves = 'regression.png'
plt_scatter = "test.png"
loss_data = 'reg_loss-ecc.csv'
# 根据输入特征和标签，自制数据集
# 导入数据集，添加数据集特征和标签路径，以及保存的路径
train_path = './单张拟合-11/train/'    # 训练集输入特征路径
train_txt = './单张拟合-11/train/regression_train.txt'  # 训练集标签txt文件
val_path = './单张拟合-11/validation/'      # 测试集输入特征路径
val_txt = './单张拟合-11/validation/regression_validation.txt'    # 测试集标签文件
print(model_name)
print("[INFO] loading dataset...")
x_train, y_train = generateds(train_path, train_txt)
x_val, y_val = generateds(val_path, val_txt)

# X_train = [cv2.cvtColor(cv2.resize(i, (256, 256)), cv2.COLOR_GRAY2RGB) for i in x_train]
# X_test = [cv2.cvtColor(cv2.resize(i, (256, 256)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(x_train).astype('float32')
x_val = np.asarray(x_val).astype('float32')
print('batch_size:', batch_size)
print(x_train.shape)
print(x_val.shape)

y_train_pd = pd.DataFrame(y_train)
y_val_pd = pd.DataFrame(y_val)
# 设置列名
y_train_pd.columns = ['label']
y_val_pd.columns = ['label']

Periodic_range = np.linspace(0, 0.750 / 2, 61)  # todo 模糊范围和模板间隔更改
y_train_pd['true_label'] = Periodic_range[y_train_pd['label']]
y_val_pd['true_label'] = Periodic_range[y_val_pd['label']]

y_train = y_train_pd['true_label']
y_val = y_val_pd['true_label']

print(x_train.shape[1:])  # (256, 256, 3)
print("[INFO] compiling model...")
# 使用VGG16模型
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])  # 第一层需要指出图像的大小
# 总参数量： 23,103,809

model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
# model.add(Activation('linear'))

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=model(base_model.output))  # VGG16模型与自己构建的模型合并

model.summary()
# 保持VGG16的前15层权值不变，即在训练过程中不训练
for layer in model.layers[:15]:
    layer.trainable = False

# initiate Adam optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
his = step_Loss()

# train the model using Adam
model.compile(loss='mse', optimizer=opt, metrics='mae')
print("[INFO] training model...")
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    shuffle=True,
                    callbacks=his)
show_lr(model).show()
print("[INFO] saving siamese model...")
model.save(model_name)
np.savetxt('reg_loss-ecc0.3.csv', his.loss, delimiter=',')  # todo
print("[INFO] plotting training history...")
reg_plot(history, loss_curves)
print('[INFO] Saved trained model at %s ' % model_name)

# # 准确率
# scores = model.evaluate(x_val, y_val, verbose=0)
# print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

predicted = model.predict(x_val)

y_pred_pd = pd.DataFrame(
    {'label': list(y_val_pd['label']), 'pred_label': list(predicted.reshape(len(predicted), ))})  # todo
y_val_pd['pred_label'] = pd.DataFrame(y_pred_pd['pred_label'])

print("[INFO] RMSE:", mean_squared_error(y_val_pd['true_label'], y_val_pd['pred_label'], squared=False))

plt.scatter(y_val_pd['true_label'], y_val_pd['pred_label'])
x = np.linspace(0, 0.750 / 2, 100)
y = x
plt.plot(x, y, color='red', linewidth=1.0, linestyle='--', label='line')
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.legend(["y = x", "pre"])
plt.title("true-pre")
plt.xlabel('true')
plt.ylabel('pre')
plt.savefig(plt_scatter, dpi=200, bbox_inches='tight', transparent=False)
plt.show()
end_time = time()
run_time = end_time - begin_time
print('[INFO] ALL TIME：', run_time, "s")  # 该循环程序运行时间
