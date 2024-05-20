#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from functools import reduce

from keras import backend as K
from keras import objectives
from keras.layers import Layer

import tensorflow as tf

from blinker import signal
import time


def _per_sample_loss(loss_function, mask, x):
    """Compute the per sample loss supporting masking and returning always
    tensor of shape (batch_size, 1)

    Arguments
    ---------
        loss_function: callable
        mask: boolean tensor or None
        x: list/tuple of inputs to the loss
    """
    # Compute the loss
    # 使用损失函数(model.loss)进行计算
    loss = loss_function(*x) # *表示解包

    # Apply masking if needed
    if mask is not None:
        if not isinstance(mask, list):
            mask = [mask]
        mask = [K.cast(m, K.floatx()) for m in mask if m is not None]
        mask = reduce(lambda a, b: a*b, mask)
        mask_dims = len(K.int_shape(mask))
        mask_mean = K.mean(mask, axis=list(range(1, mask_dims)), keepdims=True)
        loss *= mask
        loss /= mask_mean

    # If the loss has more than 1 dimensions then aggregate the last dimension
    # 如果损失的维度大于1，则聚合最后一个维度
    dims = len(K.int_shape(loss))
    if dims > 1:
        loss = K.mean(loss, axis=list(range(1, dims)))

    return K.expand_dims(loss) # 扩展维度,便于后续计算


class LossLayer(Layer):
    """LossLayer outputs the loss per sample
    
    # Arguments
        loss: The loss function to use to combine the model output and the
              target
    """
    def __init__(self, loss, **kwargs):
        self.supports_masking = True
        self.loss = objectives.get(loss)

        super(LossLayer, self).__init__(**kwargs)

    def compute_mask(self, inputs, input_mask):
        return None

    def build(self, input_shape):
        pass # Nothing to do

        super(LossLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # We need two inputs X and y
        assert len(input_shape) == 2

        # (None, 1) because all losses should be scalar
        return (input_shape[0][0], 1)

    def call(self, x, mask=None):
        return _per_sample_loss(self.loss, mask, x)


class GradientNormLayer(Layer):
    # 计算梯度范数
    """GradientNormLayer aims to output the gradient norm given a list of
    parameters (whose gradient to compute) and a loss function to combine the
    two inputs.
    
    # Arguments
        parameter_list: A list of Keras variables to compute the gradient
                        norm for
        loss: The loss function to use to combine the model output and the
              target into a scalar and then compute the gradient norm
              用于组合模型输出和真实标签，并将其转换为标量，然后计算梯度范数
        fast: If set to True it means we know that the gradient with respect to
              each sample only affects one part of the parameter list so we can
              use the batch mode to compute the gradient
    """
    def __init__(self, parameter_list, loss, fast=False, **kwargs):
        self.supports_masking = True
        self.parameter_list = parameter_list
        self.loss = objectives.get(loss)
        self.fast = fast

        super(GradientNormLayer, self).__init__(**kwargs)

    def compute_mask(self, inputs, input_mask):
        return None

    def build(self, input_shape):
        pass # Nothing to do

        super(GradientNormLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # We get two inputs
        assert len(input_shape) == 2

        return (input_shape[0][0], 1)

    def call(self, x, mask=None): # mask: 掩码，用于控制输入数据的哪些部分参与计算
        # x should be an output and a target
        # x 是预测输出与真实标签
        assert len(x) == 2
        # 计算样本loss
        losses = _per_sample_loss(self.loss, mask, x)
        if self.fast:
            # 计算梯度范数
            grads = K.sqrt(sum([
                self._sum_per_sample(K.square(g))
                for g in K.gradients(losses, self.parameter_list) # 计算梯度
            ]))
        else:
            nb_samples = K.shape(losses)[0]
            grads = K.map_fn(
                lambda i: self._grad_norm(losses[i]),
                K.arange(0, nb_samples),
                dtype=K.floatx()
            )
        # 将一维张量 grads 转换为一个一维列向量,将时间转换为张量
        return K.reshape(grads, (-1, 1))

    # 在各维度上求和，直至为一维张量
    def _sum_per_sample(self, x):
        """Sum across all the dimensions except the batch dim"""
        # Instead we might be able to use x.ndims but there have been problems
        # with ndims and Keras so I think len(int_shape()) is more reliable
        dims = len(K.int_shape(x))
        return K.sum(x, axis=list(range(1, dims)))

    def _grad_norm(self, loss):
        grads = K.gradients(loss, self.parameter_list)
        return K.sqrt(
            sum([
                K.sum(K.square(g))
                for g in grads
            ])
        )

