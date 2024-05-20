from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input
import tensorflow as tf
import time

time0 = time.time()

def debug():
    print("debug")
    """假设是计算代码"""
    time.sleep(1)
    end_time=time.time()
    return tf.convert_to_tensor(time0-time_layer,dtype=tf.float32)

def my_function():
    print("my_function")
    return model.output

# 定义输入张量
x1 = np.array([[1]])
x2 = np.array([[1,2]])
x3 = np.array([[1,2,3]])

# 创建一个简单的序贯模型
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2,name='last'))

time_layer = Input(shape=(1,))

#train = K.function([model.layers[0].input,time_layer], [model.get_output_at(0),my_function(),debug()])
train = K.function([time_layer], [debug()])

#编译模型
model.compile(optimizer='adam', loss='mse')

new_model = Model(model.input,model.output)
print("build")
#获取输出

#调用函数并打印输出
print("train")
# for i in range(10):
#     output = train([x2,time.time()])[0]
#     function= train([x2,time.time()])[1]
#     test= train([x2,time.time()])[2]
#     print("output:{}function:{}test:{}".format(output,function,test))


for i in range(10):
    start_time=time.time()
    print("start_time:{}".format(start_time))
    test= train([start_time])
    print(test)


# output: [[ 2.020913   -0.47584462  0.3191988 ]]
# function: [[-0.6344138 -0.8686867]]
# test: 1.0079327

# 使用循环调用K.function，循环体中的代码多次执行，但K.function实际只调用一次，后续调用只是获取K.function的输出