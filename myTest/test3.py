import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

# 创建一个简单的神经网络模型
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

weights1 = model.layers[0].get_weights()[0]
weights2 = model.layers[1].get_weights()[0]

# 输出第一个隐藏层的权重
print("第一个隐藏层的权重(之前):")
print(model.layers[0].get_weights()[0])

# 输出第二个隐藏层的权重
print("\n第二个隐藏层的权重(之前):")
print(model.layers[1].get_weights()[0])

# 生成随机输入数据和标签
X = np.random.randn(100, 5)
y = np.random.randint(2, size=(100, 1))

# 训练模型,会改变权重
#model.fit(X, y, epochs=1, batch_size=32) 

# 定义输入张量列表
inputs = [model.input]

# 定义输出张量列表
outputs = [layer.output for layer in model.layers]

# 使用 K.function 定义一个函数，将 inputs 作为输入，outputs 作为输出
func = K.function(inputs, outputs,) # 不传入 updates 参数
# func = K.function(inputs, outputs,updates=None) # 传入None
# updates = [] # 传入空列表
# func = K.function(inputs, outputs,updates=updates)

# 调用函数，传入随机输入数据
layer_outputs = func([X])

# 输出第一个隐藏层的权重
print("第一个隐藏层的权重(之后):")
print(model.layers[0].get_weights()[0])

# 输出第二个隐藏层的权重
print("\n第二个隐藏层的权重(之后):")
print(model.layers[1].get_weights()[0])

if(np.array_equal(weights1, model.layers[0].get_weights()[0]) and np.array_equal(weights2, model.layers[1].get_weights()[0])):
    print("权重未改变")
else:
    print("权重已改变")

# 输出
# 权重未改变