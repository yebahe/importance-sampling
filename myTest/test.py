from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras import backend as K
from keras.models import Model

import time
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


# # 获取模型的最后一层（即 Dense 层）
# dense_layer = model.layers[-1]
# print(dense_layer.get_config())
# # 修改该层的配置
# dense_layer.units = 1  # 修改神经元数量并不会更改连接权重的形状。所以模型的计算图不会受到影响，仍然会保持原来的结构。


#编译模型
model.compile(optimizer='adam', loss='mse')

new_model = Model(model.input,model.output)

#删除模型的最后一层,会影响model
model.pop()

#model.add(Dense(5))

# dense_layer_new = model.layers[-1]
# print(dense_layer_new.get_config())

#获取输出
get_hidden_output = K.function([model.layers[0].input], [model.layers[2].output])
get_hidden_output_new = K.function([new_model.layers[0].input], [new_model.layers[4].output]) # 经过Model包装后的模型,有一个额外的input层,所以同一Dense层的index+1

#调用函数并打印输出
hidden_output = get_hidden_output([x2])[0]
print(hidden_output)

hidden_output_new = get_hidden_output_new([x2])[0]
print(hidden_output_new)

# 输出
# [[ 0.23938692 -0.5397687   0.1309276 ]]
# [[ 0.52327114 -0.6721964 ]]