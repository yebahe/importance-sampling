import tensorflow as tf
from keras import backend as K
a = tf.constant(0.)
b = 2 * a
c = 2 * b
d = K.stop_gradient(2 * b)
e = a + b + c + d
gradients = K.gradients(e, [ a, b, c, d])
 
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(gradients)) 

# 输出
# [7.0, 3.0, 1.0, 1.0]
