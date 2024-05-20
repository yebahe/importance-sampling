#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras import backend as K
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Input, dot
from keras.models import Model, Sequential
import numpy as np

from importance_sampling.training import ImportanceTraining, \
    BiasedImportanceTraining, ApproximateImportanceTraining, \
    ConstantVarianceImportanceTraining, ConstantTimeImportanceTraining
from importance_sampling.samplers import BaseSampler


class TestTraining(unittest.TestCase):
    TRAININGS = [
        ImportanceTraining, BiasedImportanceTraining,
        ApproximateImportanceTraining, ConstantVarianceImportanceTraining,
        ConstantTimeImportanceTraining
    ]

    def __init__(self, *args, **kwargs):
        self.model = Sequential([
            Dense(10, activation="relu", input_shape=(2,)),
            Dense(10, activation="relu"),
            Dense(2)
        ])
        self.model.compile("sgd", "mse", metrics=["mae"])

        x1 = Input(shape=(10,))
        x2 = Input(shape=(10,))
        y = dot([
            Dense(10)(x1),
            Dense(10)(x2)
        ], axes=1)
        self.model2 = Model(inputs=[x1, x2], outputs=y)
        self.model2.compile(loss="mse", optimizer="adam")

        super(TestTraining, self).__init__(*args, **kwargs)

    def test_simple_training(self):
        for Training in self.TRAININGS:
            model = Training(self.model)
            x = np.random.rand(128, 2)
            y = np.random.rand(128, 2)

            history = model.fit(x, y, epochs=5)
            self.assertTrue("loss" in history.history)
            self.assertEqual(len(history.history["loss"]), 5)
            self.assertFalse(any(np.isnan(history.history["loss"])))

    def test_generator_training(self):
        def gen():
            while True:
                yield np.random.rand(16, 2), np.random.rand(16, 2)

        def gen2():
            while True:
                yield (np.random.rand(16, 10), np.random.rand(16, 10)), \
                    np.random.rand(16, 1)
        x_val1, y_val1 = np.random.rand(32, 2), np.random.rand(32, 2)
        x_val2, y_val2 = (np.random.rand(32, 10), np.random.rand(32, 10)), \
            np.random.rand(32, 1)

        for Training in [ImportanceTraining]:
            model = Training(self.model)
            history = model.fit_generator(
                gen(), validation_data=(x_val1, y_val1),
                steps_per_epoch=8, epochs=5
            )
            self.assertTrue("loss" in history.history)
            self.assertEqual(len(history.history["loss"]), 5)

            model = Training(self.model2)
            history = model.fit_generator(
                gen2(), validation_data=(x_val2, y_val2),
                steps_per_epoch=8, epochs=5
            )
            self.assertTrue("loss" in history.history)
            self.assertEqual(len(history.history["loss"]), 5)
            self.assertFalse(any(np.isnan(history.history["loss"])))

        with self.assertRaises(NotImplementedError):
            ApproximateImportanceTraining(self.model).fit_generator(
                gen(), validation_data=(x_val1, y_val1),
                steps_per_epoch=8, epochs=5
            )

    def test_multiple_inputs(self):
        x1 = np.random.rand(64, 10)
        x2 = np.random.rand(64, 10)
        y = np.random.rand(64, 1)

        for Training in [ImportanceTraining]:
            model = Training(self.model2)

            history = model.fit([x1, x2], y, epochs=5, batch_size=16)
            self.assertTrue("loss" in history.history)
            self.assertEqual(len(history.history["loss"]), 5)
            self.assertFalse(any(np.isnan(history.history["loss"])))

    def test_regularizers(self):
        reg = lambda w: 10
        model = Sequential([
            Dense(10, activation="relu", kernel_regularizer=reg,
                  input_shape=(2,)),
            Dense(10, activation="relu", kernel_regularizer=reg),
            Dense(2)
        ])
        model.compile("sgd", "mse")
        model = ImportanceTraining(model)
        history = model.fit(np.random.rand(64, 2), np.random.rand(64, 2))

        self.assertGreater(history.history["loss"][0], 20.)

    def test_on_sample(self):
        calls = [0]
        def on_sample(sampler, idxs, w, scores):
            calls[0] += 1
            self.assertTrue(isinstance(sampler, BaseSampler))
            self.assertEqual(len(idxs), len(w))
            self.assertEqual(len(idxs), len(scores))

        model = Sequential([
            Dense(10, activation="relu", input_shape=(2,)),
            Dense(10, activation="relu"),
            Dense(2)
        ])
        model.compile("sgd", "mse")

        for Training in self.TRAININGS:
            Training(model).fit(
                np.random.rand(64, 2), np.random.rand(64, 2),
                batch_size=16,
                epochs=4,
                on_sample=on_sample
            )
            self.assertEqual(16, calls[0])
            calls[0] = 0

    def test_metric_names(self):
        def metric1(y, y_hat):
            return K.mean((y - y_hat), axis=-1)

        model = Sequential([
            Dense(10, activation="relu", input_shape=(2,)),
            Dense(10, activation="relu"),
            Dense(2)
        ])
        model.compile("sgd", "mse", metrics=["mse", metric1])
        for Training in self.TRAININGS:
            wm = Training(model)
            self.assertEqual(
                tuple(wm.metrics_names),
                ("loss", "mse", "metric1", "score")
            )


if __name__ == "__main__":
    unittest.main()
