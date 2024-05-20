#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from functools import partial
import sys

from blinker import signal
from keras import backend as K
from keras.layers import Input, Layer, multiply
from keras.models import Model, clone_model
import numpy as np

from .layers import GradientNormLayer, LossLayer, MetricLayer
from .reweighting import UNBIASED
from .utils.functional import compose

from blinker import signal
import time


def _tolist(x, acceptable_iterables=(list, tuple)):
    if not isinstance(x, acceptable_iterables):
        x = [x]
    return x

# 依据score选定一种迭代目标
def _get_scoring_layer(score, y_true, y_pred, loss="categorical_crossentropy",
                       layer=None, model=None):
    """Get a scoring layer that computes the score for each pair of y_true,
    y_pred"""
    assert score in ["loss", "gnorm", "full_gnorm", "acc"]
    if score == "loss":
        return LossLayer(loss)([
            y_true,
            y_pred
        ])
    elif score == "gnorm":
        return GradientNormLayer(
            layer.output,
            loss,
            fast=True
        )([
            y_true,
            y_pred
        ]) # 构建对象后立即调用
    elif score == "full_gnorm":
        return GradientNormLayer(
            model.trainable_weights,
            loss,
            fast=False
        )([
            y_true,
            y_pred
        ])
    elif score == "acc":
        return LossLayer(categorical_accuracy)([
            y_true,
            y_pred
        ])


class ModelWrapper(object):
    """The goal of the ModelWrapper is to take a NN and add some extra layers
    that produce a score, a loss and the sample weights to perform importance
    sampling."""
    def _iterate_batches(self, x, y, batch_size):
        bs = batch_size
        for s in range(0, len(y), bs):
            yield [xi[s:s+bs] for xi in _tolist(x)], y[s:s+bs]

    def evaluate(self, x, y, batch_size=128):
        result = np.mean(
            np.vstack([
                self.evaluate_batch(xi, yi)
                for xi, yi in self._iterate_batches(x, y, batch_size)
            ]),
            axis=0
        )

        signal("is.evaluation").send(result)
        return result

    def score(self, x, y, batch_size=128):
        bs = batch_size
        result = np.hstack([
            self.score_batch(xi, yi).T # 计算批次中每个样本的得分
            for xi, yi in self._iterate_batches(x, y, batch_size) # 按batch_size切分数据集和标签
        ]).T
        # .T表示转置,但对于一维水平数组，.T不会改变数组的形状
        # 因此，这里的.T是为了确保返回值的一致性
        # np.hstack是水平拼接，即同行进行连接
        # result的形状是(1, batch_size)

        # 发送信号
        signal("is.score").send(result)

        return result

    def set_lr(self, lr):
        """Set the learning rate of the wrapped models.

        We try to set the learning rate on a member variable model and a member
        variable small. If we do not find a member variable model we raise a
        NotImplementedError
        """
        try:
            K.set_value(
                self.optimizer.lr,
                lr
            )
        except AttributeError:
            try:
                K.set_value(
                    self.model.optimizer.lr,
                    lr
                )
            except AttributeError:
                raise NotImplementedError()

        try:
            K.set_value(
                self.small.optimizer.lr,
                lr
            )
        except AttributeError:
            pass

    def evaluate_batch(self, x, y):
        raise NotImplementedError()

    def score_batch(self, x, y):
        raise NotImplementedError()

    def train_batch(self, x, y, w):
        raise NotImplementedError()


class ModelWrapperDecorator(ModelWrapper):
    def __init__(self, model_wrapper, implemented_attributes=set()):
        self.model_wrapper = model_wrapper
        self.implemented_attributes = (
            implemented_attributes | set(["model_wrapper"])
        )

    def __getattribute__(self, name):
        _getattr = object.__getattribute__
        implemented_attributes = _getattr(self, "implemented_attributes")
        if name in implemented_attributes:
            return _getattr(self, name)
        else:
            model_wrapper = _getattr(self, "model_wrapper")
            return getattr(model_wrapper, name)


class OracleWrapper(ModelWrapper):
    AVG_LOSS = 0
    LOSS = 1
    WEIGHTED_LOSS = 2
    SCORE = 3
    METRIC0 = 4

    FUSED_ACTIVATION_WARNING = ("[WARNING]: The last layer has a fused "
                                "activation i.e. Dense(..., "
                                "activation=\"sigmoid\").\nIn order for the "
                                "preactivation to be automatically extracted "
                                "use a separate activation layer (see "
                                "examples).\n")

    def __init__(self, model, reweighting, score="loss", layer=None):
        self.reweighting = reweighting
        self.layer = self._gnorm_layer(model, layer)

        # Augment the model with reweighting, scoring etc
        # Save the new model and the training functions in member variables
        self._augment_model(model, score, reweighting)

    def _gnorm_layer(self, model, layer):
        # If we were given a layer then use it directly
        if isinstance(layer, Layer):
            return layer

        # If we were given a layer index extract the layer
        if isinstance(layer, int):
            return model.layers[layer]

        try:
            # Get the last or the previous to last layer depending on wether
            # the last has trainable weights
            skip_one = not bool(model.layers[-1].trainable_weights)
            last_layer = -2 if skip_one else -1

            # If the last layer has trainable weights that means that we cannot
            # automatically extract the preactivation tensor so we have to warn
            # them because they might be missing out or they might not even
            # have noticed
            if last_layer == -1:
                config = model.layers[-1].get_config()
                if config.get("activation", "linear") != "linear":
                    sys.stderr.write(self.FUSED_ACTIVATION_WARNING)

            return model.layers[last_layer]
        except:
            # In case of an error then probably we are not using the gnorm
            # importance
            return None

    def _augment_model(self, model, score, reweighting):
        # Extract some info from the model
        loss = model.loss # 获取模型的损失函数
        optimizer = model.optimizer.__class__(**model.optimizer.get_config()) # 创建与原模型优化器相同的新的优化器对象
        output_shape = model.get_output_shape_at(0)[1:] # 获取模型的输出形状，不包括index0参数
        if isinstance(loss, str) and loss.startswith("sparse"): # 损失函数是稀疏损失函数
            output_shape = output_shape[:-1] + (1,) # 将最后一个维度设置为 1，以适应稀疏标签的形状。

        # Make sure that some stuff look ok
        assert not isinstance(loss, list)

        # We need to create two more inputs
        #   1. the targets
        #   2. the predicted scores
        # 创建两个新的输入层，分别用于传递真实的标签值和预测的得分值
        y_true = Input(shape=output_shape)
        pred_score = Input(shape=(reweighting.weight_size,)) # weight_size=1

        # Create a loss layer and a score layer
        # 创建损失评估层，并设定回调函数输入
        loss_tensor = LossLayer(loss)([y_true, model.get_output_at(0)])
        # 创建得分评估层，并设定回调函数输入
        score_tensor = _get_scoring_layer(
            score,
            y_true,
            model.get_output_at(0),
            loss,
            self.layer,
            model
        )

        # Create the sample weights
        # 构建权重计算层，并设定回调函数输入
        weights = reweighting.weight_layer()([score_tensor, pred_score]) # 实际上weights的内容和pred_score是一样的，但是被阻止反向传播，这意味着不会影响到模型参数的更新，也不会对模型的训练产生影响。

        # Create the output
        weighted_loss = weighted_loss_model = multiply([loss_tensor, weights]) # 将损失张量 loss_tensor 与权重张量 weights 逐元素相乘计算加权损失
        # 将所有的附加损失与加权损失相加
        for l in model.losses:
            weighted_loss += l
        weighted_loss_mean = K.mean(weighted_loss) # 计算最终加权损失均值,返回的是一个tf张量

        # Create the metric layers
        # 创建参数评估层
        metrics = model.metrics or []
        metrics = [
            MetricLayer(metric)([y_true, model.get_output_at(0)])
            for metric in metrics
        ]

        # Create a model for plotting and providing access to things such as
        # trainable_weights etc. 
        # 创建一个模型，用于绘制和提供对trainable_weights等内容的访问。
        new_model = Model(
            inputs=_tolist(model.get_input_at(0)) + [y_true, pred_score],
            outputs=[weighted_loss_model]
        )

        # Build separate on_batch keras functions for scoring and training
        # 建立单独的批内keras函数，用于评分和训练

        # 获取更新优化器
        updates = optimizer.get_updates(
            weighted_loss_mean,
            new_model.trainable_weights # 是需要进行更新（通过梯度下降）的权重列表。
        )
        
        # 可以指定更新的度量参数
        metrics_updates = []
        if hasattr(model, "metrics_updates"):
            metrics_updates = model.metrics_updates
        # 可以添加学习阶段信息
        learning_phase = []
        if weighted_loss_model._uses_learning_phase:
            learning_phase.append(K.learning_phase()) # K.learning_phase()指示当前处于训练模式还是推理模式

        # 创建输入和输出模式
        inputs = _tolist(model.get_input_at(0)) + [y_true, pred_score] + learning_phase
        outputs = [
            weighted_loss_mean,
            loss_tensor,
            weighted_loss,
            score_tensor
        ] + metrics

        # 使用Keras.function()API构建函数
        # 用于在一个 batch 上训练模型
        train_on_batch = K.function(
            inputs=inputs,
            outputs=outputs,
            updates=updates + model.updates + metrics_updates # 传入更新操作序列，会进行反向传播
        )
        # 用于在一个 batch 上评估模型
        evaluate_on_batch = K.function(
            inputs=inputs,
            outputs=outputs,
            updates=model.state_updates + metrics_updates # 没有反向传播
        )

        self.model = new_model # 扩充了功能的原本model的快照
        self.optimizer = optimizer
        self.model.optimizer = optimizer # 为model快照添加optimizer使其和原本model一样完整
        self._train_on_batch = train_on_batch
        self._evaluate_on_batch = evaluate_on_batch

    def evaluate_batch(self, x, y):
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        dummy_weights = np.ones((y.shape[0], self.reweighting.weight_size)) 
        inputs = _tolist(x) + [y, dummy_weights] + [0]
        outputs = self._evaluate_on_batch(inputs)

        # outputs[self.SCORE] = outputs[self.SCORE][:-1,:]
        signal("is.evaluate_batch").send(outputs)

        return np.hstack([outputs[self.LOSS]] + outputs[self.METRIC0:])

    def score_batch(self, x, y):
        # 执行次数= 重要性采样次数 * 预采样比例
        # 确保y为列向量
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        # dummy_weights是一个全1的数组，用于计算
        dummy_weights = np.ones((y.shape[0], self.reweighting.weight_size))
        # 复杂的高维合成数组，包含4个不同的数组对象
        inputs = _tolist(x) + [y, dummy_weights] + [0] # 0是学习率，表示不训练

        # 计算输出（包含4个不同的数组对象）
        start_time = time.time()
        outputs = self._evaluate_on_batch(inputs)
        end_time = time.time()
        signal("is.time").send(end_time - start_time,type='B_fb')

        # 返回复杂数组的score所在维度的数组对象（loss或gradient norm）
        return outputs[self.SCORE].ravel() # .ravel()将多维数组按顺序展平，返回一个水平数组

    def train_batch(self, x, y, w):
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        # train on a single batch
        start_time = time.time()
        outputs = self._train_on_batch(_tolist(x) + [y, w, 1]) # 学习率1
        end_time = time.time()
        signal("is.time").send(end_time - start_time,type='b_fb')

        # Add the outputs in a tuple to send to whoever is listening
        result = (
            outputs[self.WEIGHTED_LOSS],
            outputs[self.METRIC0:],
            outputs[self.SCORE]
        )
        signal("is.training").send(result)

        return result


class SVRGWrapper(ModelWrapper):
    """Train using SVRG."""
    def __init__(self, model):
        self._augment(model)

    def _augment(self, model):
        # TODO: There is a lot of overlap with the OracleWrapper, merge some
        #       functionality into a separate function or a parent class

        # Extract info from the model
        loss_function = model.loss
        output_shape = model.get_output_shape_at(0)[1:]

        # Create two identical models one with the current weights and one with
        # the snapshot of the weights
        self.model = model
        self._snapshot = clone_model(model)

        # Create the target variable and compute the losses and the metrics
        inputs = [
            # 创建新输入层
            Input(shape=K.int_shape(x)[1:]) # 获取输入层不包括index0参数的形状
            for x in _tolist(model.get_input_at(0)) # 获取模型的第一个输入层
        ]

        model_output = self.model(inputs) # 前向传播，得到模型的预测输出
        snapshot_output = self._snapshot(inputs) # 对照组的模型预测输出

        y_true = Input(shape=output_shape) # 创建新输入层用于传递真实的标签值，以便在训练时计算损失函数

        loss = LossLayer(loss_function)([y_true, model_output]) # 计算损失
        loss_snapshot = LossLayer(loss_function)([y_true, snapshot_output]) # 对照组的损失

        # 创建评估层
        metrics = self.model.metrics or []
        metrics = [
            MetricLayer(metric)([y_true, model_output])
            for metric in metrics
        ]

        # Make a set of variables that will be holding the batch gradient of the snapshot
        # 创建一个变量集合，用于保存对照组的批量梯度

        # 初始化了一个用于存储梯度的列表，为每个可训练参数创建一个形状相同的全零张量
        self._batch_grad = [
            K.zeros(K.int_shape(p))
            for p in self.model.trainable_weights
        ]

        # Create an optimizer that computes the variance reduced gradients and get the updates
        # 创建一个优化器，计算方差减少的梯度，并获取更新

        # 计算损失均值
        loss_mean = K.mean(loss)
        loss_snapshot_mean = K.mean(loss_snapshot)

        # 获取优化器和更新
        optimizer, updates = self._get_updates(
            loss_mean,
            loss_snapshot_mean,
            self._batch_grad
        )

        # Create the training function and gradient computation function
        metrics_updates = []
        if hasattr(self.model, "metrics_updates"):
            metrics_updates = self.model.metrics_updates
        learning_phase = []
        if loss._uses_learning_phase:
            learning_phase.append(K.learning_phase())

        #    
        inputs = inputs + [y_true] + learning_phase
        outputs = [loss_mean, loss] + metrics

        train_on_batch = K.function(
            inputs=inputs,
            outputs=outputs,
            updates=updates + self.model.updates + metrics_updates
        )
        evaluate_on_batch = K.function(
            inputs=inputs,
            outputs=outputs,
            updates=self.model.state_updates + metrics_updates
        )
        get_grad = K.function(
            inputs=inputs,
            outputs=K.gradients(loss_mean, self.model.trainable_weights),
            updates=self.model.updates
        )

        self.optimizer = optimizer
        self._train_on_batch = train_on_batch
        self._evaluate_on_batch = evaluate_on_batch
        self._get_grad = get_grad

    def _get_updates(self, loss, loss_snapshot, batch_grad):
        model = self.model
        snapshot = self._snapshot

        class Optimizer(self.model.optimizer.__class__):
            def get_gradients(self, *args):
                # 计算损失函数
                grad = K.gradients(loss, model.trainable_weights)
                grad_snapshot = K.gradients(
                    loss_snapshot,
                    snapshot.trainable_weights
                )
                return [
                    g - gs + bg
                    for g, gs, bg in zip(grad, grad_snapshot, batch_grad)
                ]

        optimizer = Optimizer(**self.model.optimizer.get_config())
        return optimizer, \
            optimizer.get_updates(loss, self.model.trainable_weights)

    def evaluate_batch(self, x, y):
        outputs = self._evaluate_on_batch(_tolist(x) + [y, 0])
        signal("is.evaluate_batch").send(outputs)

        return np.hstack(outputs[1:])

    def score_batch(self, x, y):
        raise NotImplementedError()

    def train_batch(self, x, y, w):
        outputs = self._train_on_batch(_tolist(x) + [y, 1])

        result = (
            outputs[0],   # mean loss
            outputs[2:],  # metrics
            outputs[1]    # loss per sample
        )
        signal("is.training").send(result)

        return result

    def update_grad(self, sample_generator):
        sample_generator = iter(sample_generator)
        x, y = next(sample_generator)
        N = len(y)
        gradient_sum = self._get_grad(_tolist(x) + [y, 1])
        for g_sum in gradient_sum:
            g_sum *= N
        for x, y in sample_generator:
            grads = self._get_grad(_tolist(x) + [y, 1])
            n = len(y)
            for g_sum, g in zip(gradient_sum, grads):
                g_sum += g*n
            N += len(y)
        for g_sum in gradient_sum:
            g_sum /= N

        K.batch_set_value(zip(self._batch_grad, gradient_sum))
        self._snapshot.set_weights(self.model.get_weights())


class KatyushaWrapper(SVRGWrapper):
    """Implement Katyusha training on top of plain SVRG."""
    def __init__(self, model, t1=0.5, t2=0.5):
        self.t1 = K.variable(t1, name="tau1")
        self.t2 = K.variable(t2, name="tau2")

        super(KatyushaWrapper, self).__init__(model)

    def _get_updates(self, loss, loss_snapshot, batch_grad):
        optimizer = self.model.optimizer
        t1, t2 = self.t1, self.t2
        lr = optimizer.lr

        # create copies and local copies of the parameters
        shapes = [K.int_shape(p) for p in self.model.trainable_weights]
        x_tilde = [p for p in self._snapshot.trainable_weights]
        z = [K.variable(p) for p in self.model.trainable_weights]
        y = [K.variable(p) for p in self.model.trainable_weights]

        # Get the gradients
        grad = K.gradients(loss, self.model.trainable_weights)
        grad_snapshot = K.gradients(
            loss_snapshot,
            self._snapshot.trainable_weights
        )

        # Collect the updates
        p_plus = [
            t1*zi + t2*x_tildei + (1-t1-t2)*yi
            for zi, x_tildei, yi in
            zip(z, x_tilde, y)
        ]
        vr_grad = [
            gi + bg - gsi
            for gi, bg, gsi in zip(grad, grad_snapshot, batch_grad)
        ]
        updates = [
            K.update(yi, xi - lr * gi)
            for yi, xi, gi in zip(y, p_plus, vr_grad)
        ] + [
            K.update(zi,  zi - lr * gi / t1)
            for zi, xi, gi in zip(z, p_plus, vr_grad)
        ] + [
            K.update(p, xi)
            for p, xi in zip(self.model.trainable_weights, p_plus)
        ]

        return optimizer, updates
