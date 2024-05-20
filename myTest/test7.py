import numpy as np
import time
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 生成示例数据并转换为 NumPy 数组
test = np.array(np.random.random((1000, 784)))
x_test = np.array(np.random.random((1000, 784)))
y_test = np.array(np.random.randint(2, size=(1000, 8)))

# 定义Keras函数
input_layer = model.input
output_layer = model.output
loss = K.mean(output_layer)
updates = keras.optimizers.Adam().get_updates(params=model.trainable_weights, loss=loss)
forward = K.function([input_layer, K.learning_phase()], [output_layer])
backward = K.function([input_layer, K.learning_phase()],[output_layer] ,updates)


def BuildF():
    # 编译计算图
    t1 = time.time()
    result1 = forward([x_test, 0])
    t2 = time.time()
    build_time = t2 - t1
    print("构建时间F:", build_time)

def BuildB():
    # 编译计算图
    t1 = time.time()
    result3 = backward([x_test, 0])
    t2 = time.time()
    build_time = t2 - t1
    print("构建时间FB:", build_time)

def Forward():
    # 评估正向传播的时间
    t1 = time.time()
    result4 = forward([test, 0])
    t2 = time.time()
    forward_time = t2 - t1
    print("正向传播所花费的时间：", forward_time)

def Backward():
    #评估反向传播的时间
    t1 = time.time()
    result3 = backward([test, 0])
    t2 = time.time()
    backward_time = t2 - t1
    print("反向传播所花费的时间：", backward_time)

def main(order):
    for step in order:
        if step == 1:
            BuildF()
        elif step == 2:
            BuildB()
        elif step == 3:
            Forward()
        elif step == 4:
            Backward()
        else:break

if __name__ == '__main__':
    order = [2, 1, 3, 4, 3, 4, 3, 4, 3, 4, 4, 4, 4]
    main(order)

# K.function在首次调用时会花费更多时间，因为它需要编译计算图，编译计算图会比较花费时间，即使计算图并不复杂。在后续调用中，它会更快因为只需进行计算。
# 编译计算图时，不包括updates参数时，只会编译正向传播的计算图，如果含有updates参数，会编译正向传播和反向传播的计算图。
# 因此先调用forward函数，再调用backward函数，会先编译正向传播的计算图，再编译反向传播的计算图。 而先调用backward函数，会直接编译正向传播和反向传播的计算图，再调用forward函数时只需一个非常短暂的编译。
# K.function只有在传入数据调用时才会编译计算图，而不是在定义时编译计算图。
# 计算的结果不会保存在计算图中，每次调用K.function时，都会重新计算结果，但不需要重新编译。
# backward的时间总是比forward的时间长，但不能确定是不是backward包含了forward的时间，还是只需要进行反向传播，因为更新参数也需要时间。