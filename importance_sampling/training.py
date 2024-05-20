#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import sys

from keras.callbacks import BaseLogger, CallbackList, History, ProgbarLogger
from keras.layers import Dropout, BatchNormalization
from keras.utils import Sequence
import numpy as np
from blinker import signal

from .datasets import InMemoryDataset, GeneratorDataset
from .model_wrappers import OracleWrapper, SVRGWrapper
from .samplers import ConditionalStartSampler, VarianceReductionCondition, \
    AdaptiveAdditiveSmoothingSampler, AdditiveSmoothingSampler, ModelSampler, \
    LSTMSampler, ConstantVarianceSampler, ConstantTimeSampler, SCSGSampler
from .reweighting import BiasedReweightingPolicy, NoReweightingPolicy
from .utils.functional import ___, compose, partial
import time
import pandas as pd


class _BaseImportanceTraining(object):
    def __init__(self, model, score="gnorm", layer=None):
        """Abstract base class for training a model using importance sampling.

        Arguments
        ---------
            model: The Keras model to train
        """
        # Wrap and transform the model so that it outputs the importance scores
        # and can be used in an importance sampling training scheme
        self._check_model(model)
        self.original_model = model
        # 模型包装器
        #对模型进行包装和转换，使其能输出重要性分数，并且可以用于重要性抽样训练方案
        self.model = OracleWrapper(
            model,
            self.reweighting,
            score=score,
            layer=layer
        )
        self.original_model.optimizer = self.model.optimizer
        signal("is.sample").connect(self._on_sample)
        signal("is.score").connect(self._on_scores)
    # 事件处理函数:注册了两个事件处理器，这些事件处理器将在相应的信号触发时被调用
    # 利用信号机制，这种事件驱动的编程模式可以提高代码的灵活性和可扩展性，使得不同部分的逻辑可以相互解耦，从而更容易维护和扩展
    # 当其他代码中的is.score信号被发出时 eg:signal("is.score").send(scores) ，会调用此代码中的_on_scores方法（即可以从类外部访问类内部方法）

        #计数器
        self._count = 0
        self._output_count = []
        #采样时间
        self._epoch_time_sample = 0
        self._output_epoch_time_sample = []
        #大样本计算权重和采样时间
        self._epoch_time_get_B = 0
        self._output_epoch_time_get_B = []
        #大样本网络传播时间
        self._epoch_time_B_fb = 0
        self._output_epoch_time_B_fb = []
        #小样本网络传播时间
        self._epoch_time_b_fb = 0
        self._output_epoch_time_b_fb = []
        #epoch总时间
        self._epoch_time = 0
        self._output_epoch_time = []
        #总输出
        self._output = []
        
        #信号连接
        signal("is.count").connect(self._count_)
        signal("is.time").connect(self._time_)
        
    # 使用信号机制自定义统计功能
    def _count_(self,flag=False):
        if flag:
            self._output_count.append(self._count)
            self._count = 0
        else:
            self._count += 1

    def _time_(self, t, flag=False, type=None):
        if flag:
            self._output_epoch_time_sample.append(self._epoch_time_sample)
            self._epoch_time_sample = 0
            self._output_epoch_time_get_B.append(self._epoch_time_get_B)
            self._epoch_time_get_B = 0
            self._output_epoch_time_B_fb.append(self._epoch_time_B_fb)
            self._epoch_time_B_fb = 0
            self._output_epoch_time_b_fb.append(self._epoch_time_b_fb)
            self._epoch_time_b_fb = 0
            self._output_epoch_time.append(self._epoch_time)
            self._epoch_time = 0
        else:
            if type == 'sample':
                self._epoch_time_sample += t
            elif type == 'get_B':
                self._epoch_time_get_B += t
            elif type == 'B_fb':
                self._epoch_time_B_fb += t
            elif type == 'b_fb':
                self._epoch_time_b_fb += t
            elif type == 'total':
                self._epoch_time += t
    
    # 输出到csv
    def _out_to_csv(self,output,colunms,end=False):
        df=pd.DataFrame(output,columns=colunms)
        self._output.append(df)
        if end:
            pd.concat(self._output, axis=1).to_csv('temp.csv', index=False)


    def _on_sample(self, event):
        self._latest_sample_event = event

    def _on_scores(self, scores):
        self._latest_scores = scores

    def _check_model(self, model):
        """Check if the model uses Dropout and BatchNorm and warn that it may
        affect the importance calculations."""
        if any(
            isinstance(l, (BatchNormalization, Dropout))
            for l in model.layers
        ):
            sys.stderr.write(("[NOTICE]: You are using BatchNormalization "
                              "and/or Dropout.\nThose layers may affect the "
                              "importance calculations and you are advised "
                              "to exchange them for LayerNormalization or "
                              "BatchNormalization in test mode and L2 "
                              "regularization.\n"))


    # 借用异常来暂时禁用，此方法用于定义抽象基类中的抽象属性，以强制子类实现自己的版本
    @property
    def reweighting(self):
        """The reweighting policy that controls the bias of the estimation when
        using importance sampling."""
        raise NotImplementedError()

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        """Create a new sampler to sample from the given dataset using
        importance sampling."""
        raise NotImplementedError()

    # 训练模型
    def fit(self, x, y, batch_size=32, epochs=1, verbose=1, callbacks=None,
            validation_split=0.0, validation_data=None, steps_per_epoch=None,
            on_sample=None, on_scores=None):
        """Create an `InMemoryDataset` instance with the given data and train
        the model using importance sampling for a given number of epochs.

        Arguments
        ---------
            x: Numpy array of training data or list of numpy arrays
            y: Numpy array of target data
            batch_size: int, number of samples per gradient update
            epochs: int, number of times to iterate over the entire
                    training set
            verbose: {0, >0}, whether to employ the progress bar Keras
                     callback or not
            callbacks: list of Keras callbacks to be called during training
            validation_split: float in [0, 1), percentage of data to use for
                              evaluation
            validation_data: tuple of numpy arrays, Data to evaluate the
                             trained model on without ever training on them
            steps_per_epoch: int or None, number of gradient updates before
                             considering an epoch has passed
            on_sample: callable that accepts the sampler, idxs, w, scores
            on_scores: callable that accepts the sampler and scores
        Returns
        -------
            A Keras `History` object that contains information collected during
            training.
        """
        # Create two data tuples from the given x, y, validation_*
        # 数据集处理：创建训练集和测试集

        # 直接提供了测试集
        if validation_data is not None:
            x_train, y_train = x, y
            x_test, y_test = validation_data

        # 提供了分割比例，则将原始数据集分割成训练集和测试集
        elif validation_split > 0:
            # 断言判断
            assert validation_split < 1, "100% of the data used for testing"
            n = int(round(validation_split * len(x)))
            idxs = np.arange(len(x))
            np.random.shuffle(idxs)
            x_train, y_train = x[idxs[n:]], y[idxs[n:]]
            x_test, y_test = x[idxs[:n]], y[idxs[:n]]

        # 将所有数据作为训练集，创建空的测试集
        else:
            x_train, y_train = x, y
            x_test, y_test = np.empty(shape=(0, 1)), np.empty(shape=(0, 1))

        # Make the dataset to train on
        # 创建完全在内存的数据集对象
        dataset = InMemoryDataset(
            x_train,
            y_train,
            x_test,
            y_test,
            categorical=False  # this means use the targets as is
        )

        return self.fit_dataset(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
            callbacks=callbacks,
            on_sample=on_sample,
            on_scores=on_scores
        )

    def fit_generator(self, train, steps_per_epoch=None, batch_size=32,
                      epochs=1, verbose=1, callbacks=None,
                      validation_data=None, validation_steps=None,
                      on_sample=None, on_scores=None):
        """Create a GeneratorDataset instance and train the model using
        importance sampling for a given number of epochs.

        NOTICE: This method may not be supported by all importance training
        classes and may result in NotImplementedError()

        Arguments
        ---------
            train: A generator that returns tuples (inputs, targets)
            steps_per_epoch: int, number of gradient updates before considering
                             an epoch has passed
            batch_size: int, the number of samples per gradient update (ideally
                             set to the number of items returned by the
                             generator at each call)
            epochs: int, multiplied by steps_per_epoch denotes the total number
                    of gradient updates
            verbose: {0, >0}, whether to use the progress bar Keras callback
            validation_data: generator or tuple (inputs, targets)
            validation_steps: None or int, used only if validation_data is a
                              generator
            on_sample: callable that accepts the sampler, idxs, w, scores
            on_scores: callable that accepts the sampler and scores
        """
        def sequence_gen(seq):
            i = 0
            while True:
                yield seq[i]
                i = (i+1) % len(seq)

        # Create the validation data to pass to the GeneratorDataset
        if validation_data is not None:
            if isinstance(validation_data, (tuple, list)):
                test = validation_data
                test_len = None
            elif isinstance(validation_data, Sequence):
                test = sequence_gen(validation_data)
                test_len = len(validation_data) * batch_size
            else:
                test = validation_data
                test_len = validation_steps * batch_size
        else:
            test = (np.empty(shape=(0, 1)), np.empty(shape=(0, 1)))
            test_len = None

        if isinstance(train, Sequence):
            if steps_per_epoch is None:
                steps_per_epoch = len(train)
            train = sequence_gen(train)
        elif steps_per_epoch is None:
            raise ValueError("steps_per_epoch is not set")

        dataset = GeneratorDataset(train, test, test_len)

        return self.fit_dataset(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
            callbacks=callbacks,
            on_sample=on_sample,
            on_scores=on_scores
        )

    def fit_dataset(self, dataset, steps_per_epoch=None, batch_size=32,
                    epochs=1, verbose=1, callbacks=None, on_sample=None,
                    on_scores=None):
        """Train the model on the given dataset for a given number of epochs.

        Arguments
        ---------
            dataset: Instance of `BaseDataset` that provides the data
                     to train on.
            steps_per_epoch: int or None, number of gradient updates before
                             considering an epoch has passed. If None it is set
                             to be `len(dataset.train_data) / batch_size`.
            batch_size: int, number of samples per gradient update
            epochs: int, number of times to iterate `steps_per_epoch` times
            verbose: {0, >0}, whether to employ the progress bar Keras
                     callback or not
            callbacks: list of Keras callbacks to be called during training
            on_sample: callable that accepts the sampler, idxs, w, scores
            on_scores: callable that accepts the sampler and scores
        """
        try:
            if len(dataset.train_data) < batch_size:
                raise ValueError(("The model cannot be trained with "
                                  "batch_size > training set"))
        except RuntimeError as e:
            assert "no size" in str(e)

        # Set steps_per_epoch properly
        if steps_per_epoch is None:
            steps_per_epoch = len(dataset.train_data) // batch_size # //:地板除，返回不大于结果的最大整数

        # Create the callbacks list
        # 创建回调函数列表，主要作用是在训练过程中记录训练的一些信息
        self.history = History()
        callbacks = [BaseLogger()] + (callbacks or []) + [self.history]
        if verbose > 0:
            callbacks += [ProgbarLogger(count_mode="steps")]
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self.original_model)
        callbacks.set_params({
            "epochs": epochs,
            "steps": steps_per_epoch,
            "verbose": verbose,
            "do_validation": len(dataset.test_data) > 0,
            "metrics": self.metrics_names + [
                "val_" + n for n in self.metrics_names
            ]
        })

        # Create the sampler
        # 创建采样器
        sampler = self.sampler(dataset, batch_size, steps_per_epoch, epochs)

        # Start the training loop
        # 训练
        epoch = 0
        self.original_model.stop_training = False
        callbacks.on_train_begin()
        while epoch < epochs:
            callbacks.on_epoch_begin(epoch)
            # 数据量较大时，将一个epoch分为多个step进行
            for step in range(steps_per_epoch):
                batch_logs = {"batch": step, "size": batch_size}
                callbacks.on_batch_begin(step, batch_logs)

                # Importance sampling is done here
                # 重要性采样
                # idxs:被重要性采样选中的样本索引
                # x:输入数据，y:标签数据，来自原始的训练数据中被随机选中的大批量样本数据
                # w是通过重要性采样后计算出被抽样样本用于SGD的权重，而非用于重要性采样的权重

                time1 = time.time()
                idxs, (x, y), w = sampler.sample(batch_size) #不包含反向传播
                time2 = time.time()

                # Train on the sampled data
                # 迭代
                loss, metrics, scores = self.model.train_batch(x, y, w) # 包含反向传播，更新参数

                # Update the sampler
                # 更新采样器参数τ
                sampler.update(idxs, scores)
                time3 = time.time()

                signal("is.time").send(time2-time1,type='sample')
                signal("is.time").send(time3-time1,type='total')

                values = map(lambda x: x.mean(), [loss] + metrics)
                for l, o in zip(self.metrics_names, values):
                    batch_logs[l] = o
                callbacks.on_batch_end(step, batch_logs)

                # 如果传入了on_sample和on_scores回调函数，则在每个batch结束后调用
                # 用于记录每个batch的采样和重要性分数
                if on_scores is not None and hasattr(self, "_latest_scores"):
                    on_scores(
                        sampler,
                        self._latest_scores
                    )
                if on_sample is not None:
                    on_sample(
                        sampler,
                        self._latest_sample_event["idxs"],
                        self._latest_sample_event["w"],
                        self._latest_sample_event["predicted_scores"]
                    )

                if self.original_model.stop_training:
                    break

            # Evaluate now that an epoch passed
            epoch_logs = {}
            if len(dataset.test_data) > 0:
                val = self.model.evaluate(
                    *dataset.test_data[:],
                    batch_size=batch_size
                )
                epoch_logs = {
                    "val_" + l: o
                    for l, o in zip(self.metrics_names, val)
                }
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.original_model.stop_training:
                break
            epoch += 1

            signal("is.count").send(True)
            signal("is.time").send(flag=True)

        self._out_to_csv(self._output_count,['sample_number'])
        self._out_to_csv(self._output_epoch_time_sample,['sample_time'])
        self._out_to_csv(self._output_epoch_time_get_B,['get_B_time'])
        self._out_to_csv(self._output_epoch_time_B_fb,['B_f_time'])
        self._out_to_csv(self._output_epoch_time_b_fb,['b_fb_time'])
        self._out_to_csv(self._output_epoch_time,['epoch_time'],end=True)

        callbacks.on_train_end()

        return self.history

    @property
    def metrics_names(self):
        def name(x):
            if isinstance(x, str):
                return x
            if hasattr(x, "name"):
                return x.name
            if hasattr(x, "__name__"):
                return x.__name__
            return str(x)
        metrics = self.original_model.metrics or []
        return (
            ["loss"] +
            [name(m) for m in metrics] +
            ["score"]
        )


class _UnbiasedImportanceTraining(_BaseImportanceTraining):
    def __init__(self, model, score="gnorm", layer=None):
        self._reweighting = BiasedReweightingPolicy(1.0)  # no bias

        super(_UnbiasedImportanceTraining, self).__init__(model, score, layer)

    # 装饰器：将方法转化为只读属性以提供访问
    @property
    # 提供对重要性重加权策略的访问
    def reweighting(self):
        return self._reweighting


class ImportanceTraining(_UnbiasedImportanceTraining):
    """Train a model with exact importance sampling.

    Arguments
    ---------
        model: The Keras model to train
        presample: float, the number of samples to presample for scoring
                   given as a factor of the batch size
        tau_th: float or None, the variance reduction threshold after which we
                enable importance sampling
        forward_batch_size: int or None, the batch size to use for the forward
                            pass during scoring
        score: {"gnorm", "loss", "full_gnorm"}, the importance metric to use
               for importance sampling
        layer: None or int or Layer, the layer to compute the gnorm with
    """
    def __init__(self, model, presample=3.0, tau_th=None,
                 forward_batch_size=None, score="gnorm", layer=None):
        self._presample = presample # 采样比例 = large_batch / batch_size
        self._tau_th = tau_th
        self._forward_batch_size = forward_batch_size

        # Call the parent to wrap the model
        # 调用父类包装模型
        super(ImportanceTraining, self).__init__(model, score, layer)

    # 配置采样器
    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        # Configure the sampler
        large_batch = int(self._presample * batch_size)
        forward_batch_size = self._forward_batch_size or batch_size
        # Compute the threshold using eq. 29 in
        # https://arxiv.org/abs/1803.00942
        B = large_batch
        b = batch_size
        tau_th = float(B + 3*b) / (3*b)
        tau_th = self._tau_th if self._tau_th is not None else tau_th

        return ConditionalStartSampler(
            ModelSampler(
                dataset,
                self.reweighting,
                self.model,
                large_batch=large_batch,
                forward_batch_size=forward_batch_size
            ),
            VarianceReductionCondition(tau_th)
        )


class ConstantVarianceImportanceTraining(_UnbiasedImportanceTraining):
    """Train a model faster by keeping the per iteration variance constant but
    decreasing the time.

    Arguments
    ---------
        model: The Keras model to train
        score: {"gnorm", "loss", "full_gnorm"}, the importance metric to use
               for importance sampling
        layer: None or int or Layer, the layer to compute the gnorm with
    """
    def __init__(self, model, backward_time=2.0, extra_samples=0.2,
                 score="gnorm", layer=None):
        self._backward_time = backward_time
        self._extra_samples = extra_samples

        super(ConstantVarianceImportanceTraining, self).__init__(
            model,
            score,
            layer
        )

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        return ConstantVarianceSampler(
            dataset,
            self.reweighting,
            self.model,
            backward_time=self._backward_time,
            extra_samples=self._extra_samples
        )


class ConstantTimeImportanceTraining(_UnbiasedImportanceTraining):
    """Train a model faster by keeping the per iteration time constant but
    improving the quality of the gradients.

    Arguments
    ---------
        model: The Keras model to train
        score: {"gnorm", "loss", "full_gnorm"}, the importance metric to use
               for importance sampling
        layer: None or int or Layer, the layer to compute the gnorm with
    """
    def __init__(self, model, backward_time=2.0, score="gnorm", layer=None):
        self._backward_time = backward_time

        super(ConstantTimeImportanceTraining, self).__init__(
            model,
            score,
            layer
        )

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        return ConstantTimeSampler(
            dataset,
            self.reweighting,
            self.model,
            backward_time=self._backward_time
        )


class BiasedImportanceTraining(_BaseImportanceTraining):
    """Train a model with exact importance sampling using the loss as
    importance.

    Arguments
    ---------
        model: The Keras model to train
        k: float in (-oo, 1], controls the bias of the training that focuses
           the network on the hard examples (see paper)
        smooth: float, influences the sampling distribution towards uniform by
                adding `smooth` to all probabilities and renormalizing
        adaptive_smoothing: bool, If True the `smooth` argument is a percentage
                            of the average training loss
        presample: int, the number of samples to presample for scoring
        forward_batch_size: int, the batch size to use for the forward pass
                            during scoring
    """
    def __init__(self, model, k=0.5, smooth=0.0, adaptive_smoothing=False,
                 presample=256, forward_batch_size=128):
        # Create the reweighting policy
        self._reweighting = BiasedReweightingPolicy(k)

        # Call the parent to wrap the model
        super(BiasedImportanceTraining, self).__init__(model)

        # Create the sampler factory, the workhorse of the whole deal :-)
        adaptive_smoothing_factory = partial(
            AdaptiveAdditiveSmoothingSampler, ___, smooth
        )
        additive_smoothing_factory = partial(
            AdditiveSmoothingSampler, ___, smooth
        )
        self._sampler = partial(
            ModelSampler,
            ___,
            self.reweighting,
            self.model,
            large_batch=presample,
            forward_batch_size=forward_batch_size
        )
        if adaptive_smoothing and smooth > 0:
            self._sampler = compose(adaptive_smoothing_factory, self._sampler)
        elif smooth > 0:
            self._sampler = compose(additive_smoothing_factory, self._sampler)

    @property
    def reweighting(self):
        return self._reweighting

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        return self._sampler(dataset)


class ApproximateImportanceTraining(_BaseImportanceTraining):
    """Train a model with importance sampling using an LSTM with a class
    embedding to predict the importance of the training samples.

    Arguments
    ---------
        model: The Keras model to train
        k: float in (-oo, 1], controls the bias of the training that focuses
           the network on the hard examples (see paper)
        smooth: float, influences the sampling distribution towards uniform by
                adding `smooth` to all probabilities and renormalizing
        adaptive_smoothing: bool, If True the `smooth` argument is a percentage
                            of the average training loss
        presample: int, the number of samples to presample for scoring
    """
    def __init__(self, model, k=0.5, smooth=0.0, adaptive_smoothing=False,
                 presample=2048):
        # Create the reweighting policy
        self._reweighting = BiasedReweightingPolicy(k)

        # Call the parent to wrap the model
        super(ApproximateImportanceTraining, self).__init__(model)

        # Create the sampler factory, the workhorse of the whole deal :-)
        adaptive_smoothing_factory = partial(
            AdaptiveAdditiveSmoothingSampler, ___, smooth
        )
        additive_smoothing_factory = partial(
            AdditiveSmoothingSampler, ___, smooth
        )
        self._sampler = partial(
            LSTMSampler,
            ___,
            self.reweighting,
            presample=presample
        )
        if adaptive_smoothing and smooth > 0:
            self._sampler = compose(adaptive_smoothing_factory, self._sampler)
        elif smooth > 0:
            self._sampler = compose(additive_smoothing_factory, self._sampler)

    @property
    def reweighting(self):
        return self._reweighting

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        return self._sampler(dataset)

    def fit_generator(*args, **kwargs):
        raise NotImplementedError("ApproximateImportanceTraining doesn't "
                                  "support generator training")


class SVRG(_BaseImportanceTraining):
    """Train a model with Stochastic Variance Reduced Gradient descent.

    See [1, 2] for what this trainer implements.

    [1] Johnson, R. and Zhang, T., 2013. Accelerating stochastic gradient
        descent using predictive variance reduction. In Advances in neural
        information processing systems (pp. 315-323).
    [2] Lei, L. and Jordan, M., 2017, April. Less than a Single Pass:
        Stochastically Controlled Stochastic Gradient. In Artificial
        Intelligence and Statistics (pp. 148-156).

    Arguments
    ---------
        model: The Keras model to train
        B: float, the number of samples to use for estimating the average
           gradient (whole dataset used in case of 0). Given as a factor of the
           batch_size.
        B_rate: float, multiply B with B_rate after every update
        B_over_b: int, How many updates to perform before recalculating the
                  average gradient
    """
    def __init__(self, model, B=10., B_rate=1.0, B_over_b=128):
        self._B = B
        self._B_rate = B_rate
        self._B_over_b = B_over_b

        self.original_model = model
        self.model = SVRGWrapper(model)

    @property
    def reweighting(self):
        """SVRG does not need sample weights so it returns the
        NoReweightingPolicy()"""
        return NoReweightingPolicy()

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        """Create the SCSG sampler"""
        B = int(self._B * batch_size)
        
        return SCSGSampler(
            dataset,
            self.reweighting,
            self.model,
            B,
            self._B_over_b,
            self._B_rate
        )
