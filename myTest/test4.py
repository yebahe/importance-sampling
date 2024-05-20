import time
from keras.callbacks import Callback

import numpy as np

# 生成训练数据
def generate_data(num_samples):
    X = np.random.rand(num_samples, 5)  # 特征维度为5
    y = np.random.randint(2, size=num_samples)  # 生成随机的二分类标签
    return X, y

# 生成100个样本作为示例
X_train, y_train = generate_data(100)

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time.time() - self.epoch_start_time
        self.times.append(epoch_time)
        print("Epoch %d - Time: %.2fs" % (epoch, epoch_time))

# 使用示例
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape=(5,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

# 创建并添加时间记录回调
time_callback = TimeHistory()
model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[time_callback])
print(time_callback.times)

# 输出
# Epoch 1/10
# 2024-04-02 22:55:29.970134: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX 
# AVX2
# 2024-04-02 22:55:29.971913: I tensorflow/core/common_runtime/process_util.cc:71] Creating new thread pool with default inter op setting: 16. Tune using inter_op_parallelism_threads for best performance.
# 100/100 [==============================] - 0s 3ms/step - loss: 0.7963
# Epoch 0 - Time: 0.28s
# Epoch 2/10
# 100/100 [==============================] - 0s 30us/step - loss: 0.7891
# Epoch 1 - Time: 0.00s
# Epoch 3/10
# 100/100 [==============================] - 0s 30us/step - loss: 0.7810
# Epoch 2 - Time: 0.00s
# Epoch 4/10
# 100/100 [==============================] - 0s 30us/step - loss: 0.7753
# Epoch 3 - Time: 0.00s
# Epoch 5/10
# 100/100 [==============================] - 0s 30us/step - loss: 0.7684
# Epoch 4 - Time: 0.00s
# Epoch 6/10
# 100/100 [==============================] - 0s 30us/step - loss: 0.7627
# Epoch 5 - Time: 0.00s
# Epoch 7/10
# 100/100 [==============================] - 0s 40us/step - loss: 0.7572
# Epoch 6 - Time: 0.00s
# Epoch 8/10
# 100/100 [==============================] - 0s 30us/step - loss: 0.7529
# Epoch 7 - Time: 0.00s
# Epoch 9/10
# 100/100 [==============================] - 0s 30us/step - loss: 0.7467
# Epoch 8 - Time: 0.00s
# Epoch 10/10
# 100/100 [==============================] - 0s 30us/step - loss: 0.7416
# Epoch 9 - Time: 0.00s
# [0.2802093029022217, 0.0029997825622558594, 0.004000186920166016, 0.003000020980834961, 0.003000020980834961, 0.0029997825622558594, 0.003999948501586914, 0.004000186920166016, 0.003000497817993164, 0.0029997825622558594]