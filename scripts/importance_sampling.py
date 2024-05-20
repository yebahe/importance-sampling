#!/usr/bin/env python
#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""This script aims to be the single experiment script for fast prototyping of
importance sampling methods"""

from __future__ import print_function

import argparse
from bisect import bisect_right
from contextlib import contextmanager
from itertools import product
import os
from os import path
import time

from blinker import signal
import numpy as np
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import plot_model

from importance_sampling import models
from importance_sampling.datasets import CIFAR10, CIFAR100, CIFARSanityCheck, \
    CanevetICML2016, MNIST, OntheflyAugmentedImages, PennTreeBank, \
    ImageNetDownsampled, TIMIT, ZCAWhitening, MIT67, CASIAWebFace, \
    CASIAWebFace2, PixelByPixelMNIST
from importance_sampling.reweighting import AdjustedBiasedReweightingPolicy, \
    BiasedReweightingPolicy, NoReweightingPolicy, CorrectingReweightingPolicy
from importance_sampling.model_wrappers import OracleWrapper, SVRGWrapper, \
    KatyushaWrapper
from importance_sampling.samplers import ModelSampler, UniformSampler, \
    LSTMSampler, PerClassGaussian, LSTMComparisonSampler, \
    AdditiveSmoothingSampler, AdaptiveAdditiveSmoothingSampler, \
    PowerSmoothingSampler, OnlineBatchSelectionSampler, HistorySampler, \
    CacheSampler, ConditionalStartSampler, WarmupCondition, ExpCondition, \
    TotalVariationCondition, VarianceReductionCondition, SCSGSampler
from importance_sampling.utils import tf_config
from importance_sampling.utils.functional import compose, partial, ___


@contextmanager
def ignored(exception_class):
    """Ignore exceptions of the passed as argument type."""
    try:
        yield
    except exception_class:
        pass


class WallClock(object):
    """A class to write the elapsed wall clock time in a file"""
    def __init__(self, filepath):
        self._file = open(filepath, "w")
        self._start = time.time()

    @property
    def elapsed(self):
        return time.time() - self._start

    def write(self):
        self._file.write("%f\n" % (self.elapsed,))
        self._file.flush()


def create_dict(x):
    """Create a typed dictionary of simple values (strings, floats, ints) for
    passing arbitrary parameters to the script"""
    def array(t):
        def inner(x):
            return list(map(t, x.split("!")))
        return inner

    types = {
        "s": str,        "f": float,        "i": int,
        "S": array(str), "F": array(float), "I": array(int)
    }
    d = {}
    if x:
        for kv in x.split(";"):
            key, value = kv.split("=", 1)
            d[key] = types[value[0]](value[1:])

    return d


def iterate_grid(grid):
    def items(param):
        for x in param[1]:
            yield param[0], x
    return product(*list(map(items, grid)))


def name_from_grid(values, prefix="", sep="-"):
    def tostr(x):
        x = x[1]
        try:
            if int(x) == x:
                return str(int(x))
        except ValueError:
            pass
        return str(x)
    return sep.join(
        ([prefix] if prefix else []) +
        list(map(tostr, values))
    )


def first_or_self(x):
    try:
        return x[0]
    except:
        return x


def load_dataset(dataset, hyperparams):
    datasets = {
        "canevet-icml2016-jittered": partial(CanevetICML2016, smooth=10),
        "canevet-icml2016": CanevetICML2016,
        "canevet-icml2016-smooth": partial(CanevetICML2016, smooth=15000),
        "cifar-sanity-check": CIFARSanityCheck,
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "cifar10-augmented": compose(
            partial(OntheflyAugmentedImages, ___, dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False
            )),
            CIFAR10
        ),
        "cifar10-whitened-augmented": compose(
            partial(OntheflyAugmentedImages, ___, dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False
            ), N=15*10**5),
            ZCAWhitening,
            CIFAR10
        ),
        "cifar100-augmented": compose(
            partial(OntheflyAugmentedImages, ___, dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False
            )),
            CIFAR100
        ),
        "cifar100-whitened-augmented": compose(
            partial(OntheflyAugmentedImages, ___, dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False
            ), N=15*10**5),
            ZCAWhitening,
            CIFAR100
        ),
        "ptb": partial(PennTreeBank, 20),
        "imagenet-32x32": partial(
            ImageNetDownsampled,
            hyperparams.get("imagenet", os.getenv("IMAGENET")),
            size=32
        ),
        "imagenet-64x64": partial(
            ImageNetDownsampled,
            hyperparams.get("imagenet", os.getenv("IMAGENET")),
            size=64
        ),
        "timit": partial(
            TIMIT,
            20,
            hyperparams.get("timit", os.getenv("TIMIT"))
        ),
        "mit-67": partial(
            MIT67,
            hyperparams.get("mit67", os.getenv("MIT67"))
        ),
        "casia": partial(
            CASIAWebFace,
            hyperparams.get("casia", os.getenv("CASIA")),
            alpha=hyperparams.get("triplet_alpha", 0.2),
            embedding=hyperparams.get("triplet_embedding", 128)
        ),
        "casia-cls": partial(
            CASIAWebFace2,
            hyperparams.get("casia", os.getenv("CASIA")),
        ),
        "pixel-mnist": PixelByPixelMNIST
    }

    return datasets[dataset]()


def build_reweighting_policy(policy, hyperparams={}):
    policies = {
        "unweighted": NoReweightingPolicy,
        "predicted": partial(
            BiasedReweightingPolicy,
            k=hyperparams.get("k", 1.0)
        ),
        "adjusted": partial(
            AdjustedBiasedReweightingPolicy,
            k=hyperparams.get("k", 1.0)
        ),
        "correcting": partial(
            CorrectingReweightingPolicy,
            k=hyperparams.get("k", 1.0)
        )
    }

    return policies[policy]()


def get_models_dictionary(hyperparams={}, reweighting=None):
    classes = {
        "oracle": partial(OracleWrapper, ___, reweighting),
    }

    grids = {
        "oracle": [
            ("score", ["loss", "gnorm", "acc"])
        ],
    }

    wrappers = {}
    for k in classes:
        for items in iterate_grid(grids[k]):
            final_key = name_from_grid(items, prefix=k)
            wrappers[final_key] = partial(classes[k], **dict(items))

    wrappers["svrg"] = SVRGWrapper
    wrappers["katyusha"] = partial(
        KatyushaWrapper,
        ___,
        t1=hyperparams.get("kat_t1", 0.5),
        t2=hyperparams.get("kat_t2", 0.5)
    )

    return wrappers


def build_model(model, wrapper, dataset, hyperparams, reweighting):
    def build_optimizer(opt, hyperparams):
        return {
            "sgd": SGD(
                lr=hyperparams.get("lr", 0.001),
                momentum=hyperparams.get("momentum", 0.0),
                nesterov=hyperparams.get("nesterov", False),
                clipnorm=hyperparams.get("clipnorm", 0)
            ),
            "adam": Adam(
                lr=hyperparams.get("lr", 0.001),
                clipnorm=hyperparams.get("clipnorm", 0)
            ),
            "rmsprop": RMSprop(
                lr=hyperparams.get("lr", 0.001),
                decay=hyperparams.get("lr_decay", 0.0),
                clipnorm=hyperparams.get("clipnorm", 0)
            )
        }[opt]

    model = models.get(model)(dataset.shape, dataset.output_size)
    model.compile(
        optimizer=build_optimizer(
            hyperparams.get("opt", "adam"),
            hyperparams
        ),
        loss=model.loss,
        metrics=model.metrics
    )

    return get_models_dictionary(hyperparams, reweighting)[wrapper](model)


def get_samplers_dictionary(model, hyperparams={}, reweighting=None):
    # Add some base samplers
    samplers = {
        "uniform": partial(UniformSampler, ___, reweighting),
        "model": partial(
            ModelSampler,
            ___,
            reweighting,
            model,
            large_batch=hyperparams.get("presample", 1024)
        ),
        "lstm": partial(
            LSTMSampler,
            ___,
            reweighting,
            log=hyperparams.get("lstm_log", 0) != 0,
            presample=hyperparams.get("presample", 2048)
        ),
        "pcg": partial(
            PerClassGaussian,
            ___,
            reweighting,
            alpha=hyperparams.get("alpha", 0.9),
            presample=hyperparams.get("presample", 2048)
        ),
        "obs": partial(
            OnlineBatchSelectionSampler,
            ___,
            reweighting,
            model,
            steps_per_epoch=hyperparams.get("steps_per_epoch", 300),
            recompute=hyperparams.get("obs_recompute", 2),
            s_e=(
                hyperparams.get("obs_se0", 1),
                hyperparams.get("obs_seend", 1)
            ),
            n_epochs=hyperparams.get("obs_epochs", 1)
        ),
        "history": partial(
            HistorySampler,
            ___,
            reweighting,
            model,
            recompute=hyperparams.get("history_recompute", 600)
        ),
        "cache": partial(
            CacheSampler,
            ___,
            reweighting,
            staleness=hyperparams.get("staleness", 3),
            cache_prob=hyperparams.get("cache_prob", 0.5),
            smooth=hyperparams.get("smooth", 0.2)
        ),
        "scsg": partial(
            SCSGSampler,
            ___,
            reweighting,
            model,
            B=hyperparams.get("scsg_B", 0),
            B_over_b=hyperparams.get("scsg_B_over_b", 1000),
            B_rate=hyperparams.get("scsg_B_rate", 1.0)
        )
    }

    samplers["lstm-comparison"] = lambda x: LSTMComparisonSampler(
        x,
        samplers["lstm"](x),
        samplers["model"](x),
        subset=hyperparams.get("lstm_comparison", 1024)
    )

    # Add some decorated samplers
    samplers_for_decoration = [
        "model",
        "lstm",
        "pcg",
        "history"
    ]
    for sampler in samplers_for_decoration:
        samplers["smooth-"+sampler] = compose(
            partial(
                AdditiveSmoothingSampler,
                c=hyperparams.get("smooth", 1.0)
            ),
            samplers[sampler]
        )
        samplers["adaptive-smooth-"+sampler] = compose(
            partial(
                AdaptiveAdditiveSmoothingSampler,
                percentage=hyperparams.get("smooth", 0.5)
            ),
            samplers[sampler]
        )
        samplers["power-smooth-"+sampler] = compose(
            partial(
                PowerSmoothingSampler,
                power=hyperparams.get("smooth", 0.5)
            ),
            samplers[sampler]
        )

    for sampler in list(samplers.keys()):
        if "model" in sampler or "lstm" in sampler:
            samplers["warmup-"+sampler] = compose(
                partial(
                    ConditionalStartSampler,
                    ___,
                    WarmupCondition(hyperparams.get("warmup", 100))
                ),
                samplers[sampler]
            )
            samplers["exp-"+sampler] = compose(
                partial(
                    ConditionalStartSampler,
                    ___,
                    ExpCondition(hyperparams.get("exp_th", 2.0))
                ),
                samplers[sampler]
            )
            samplers["tv-"+sampler] = compose(
                partial(
                    ConditionalStartSampler,
                    ___,
                    TotalVariationCondition(hyperparams.get("tv_th", 0.5))
                ),
                samplers[sampler]
            )
            samplers["vr-"+sampler] = compose(
                partial(
                    ConditionalStartSampler,
                    ___,
                    VarianceReductionCondition(hyperparams.get("vr_th", 0.5))
                ),
                samplers[sampler]
            )

    return samplers


def build_sampler(sampler, dataset, model, hyperparams, reweighting):
    return get_samplers_dictionary(
        model,
        hyperparams,
        reweighting
    )[sampler](dataset)


class PredictionLogger(object):
    """Log the samples with the respective actual and predicted metrics"""
    def __init__(self, output_directory, filename):
        self.fd = open(path.join(output_directory, filename), "w")

    def on_sample(self, sample):
        self._y = sample["xy"][1].argmax(axis=1)
        self._scores_hat = sample["predicted_scores"]

    def on_training(self, metrics):
        if self._scores_hat is None:
            return
        self._scores = metrics[2].ravel()
        for s, s_hat, y in zip(self._scores, self._scores_hat, self._y):
            print(s, s_hat, y, file=self.fd)
        self.fd.flush()


class EvaluationLogger(object):
    """Log the results of the evaluation during training"""
    def __init__(self, output_directory, test, test_scores,
                 train, train_scores):
        # Complete filenames
        test = path.join(output_directory, test)
        test_scores = path.join(output_directory, test_scores)
        train = path.join(output_directory, train)
        train_scores = path.join(output_directory, train_scores)

        # Open the files
        self.fd_test = open(test, "w")
        self.fd_test_scores = open(test_scores, "w")
        self.fd_train = open(train, "w")
        self.fd_train_scores = open(train_scores, "w")

        # Write the headers
        print("Crossent", "Accuracy", file=self.fd_test)
        print("Crossent", "Accuracy", file=self.fd_train)

        # Variable that tells us to which file we are writing
        self._test = True

    def save_test(self):
        self._test = True

    def save_train(self):
        self._test = False

    def on_evaluate(self, metrics):
        fd = self.fd_test if self._test else self.fd_train
        print(first_or_self(metrics[0]), first_or_self(metrics[1]), file=fd)
        fd.flush()

    def on_batch(self, metrics):
        fd = self.fd_test_scores if self._test else self.fd_train_scores
        for score in metrics[-1].ravel():
            print(score, file=fd)
        fd.flush()


def metrics_progress(output_directory, filename):
    """Create a function that logs the first loss and metric in a file from a
    metrics event

    We are leaking a file descriptor but its lifetime would be as long as the
    program anyway
    """
    fd = open(path.join(output_directory, filename), "w")
    print("Crossent", "Accuracy", file=fd)
    def inner(metrics):
        losses, metrics = metrics[:2]
        loss = losses.mean()
        acc = metrics[0].mean()
        print(loss, acc, file=fd)
        fd.flush()
    return inner


def log_lines(output_directory, filename, values_slice=slice(None),
              initial=[], multiple=False):
    """Log the values passed in as a parameter to a file as space separated"""
    fd = open(path.join(output_directory, filename), "w")
    if initial:
        print(" ".join(list(map(str, initial))), file=fd)
    def inner(values):
        if not multiple:
            values = [values]
        for v in values:
            print(" ".join(list(map(str, v[values_slice]))), file=fd)
        fd.flush()
    return inner


def every_nth(n):
    i = [0]
    def inner():
        i[0] += 1
        return i[0] % n == 0
    return inner


def main(argv):
    parser = argparse.ArgumentParser(
        description="Perform importance sampling experiments"
    )

    parser.add_argument(
        "model",
        choices=[
            "small_nn", "small_cnn", "cnn", "elu_cnn", "lstm_lm", "lstm_lm2",
            "lstm_lm3", "small_cnn_sq", "wide_resnet_16_4", "wide_resnet_28_10",
            "wide_resnet_28_2", "wide_resnet_16_4_dropout",
            "wide_resnet_28_10_dropout", "lstm_timit", "pretrained_resnet50",
            "triplet_pre_densenet121", "pretrained_densenet121",
            "triplet_pre_resnet50", "face_pre_resnet50", "lstm_mnist",
            "svrg_nn"
        ],
        help="Choose the NN model to build"
    )
    parser.add_argument(
        "wrapper",
        choices=list(get_models_dictionary().keys()),
        help="Choose what to wrap the model with to compute importance scores"
    )
    parser.add_argument(
        "sampler",
        choices=list(get_samplers_dictionary(None).keys()),
        help="Choose the sampler to draw training data"
    )
    parser.add_argument(
        "reweighting",
        choices=["unweighted", "predicted", "adjusted", "correcting"],
        help="Choose the sample weight policy"
    )
    parser.add_argument(
        "dataset",
        choices=[
            "canevet-icml2016-jittered", "canevet-icml2016",
            "canevet-icml2016-smooth", "cifar-sanity-check", "mnist",
            "cifar10", "cifar100", "cifar10-augmented", "cifar100-augmented",
            "ptb", "cifar10-whitened-augmented", "imagenet-32x32",
            "imagenet-64x64", "timit", "cifar100-whitened-augmented",
            "mit-67", "casia", "casia-cls", "pixel-mnist"
        ],
        help="Choose the dataset to train on"
    )
    parser.add_argument(
        "output_directory",
        help="The directory in which to save the output"
    )

    parser.add_argument(
        "--train_for",
        type=int,
        default=10000,
        help="Stop training after seeing that many mini-batches"
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=300,
        help="Compute the validation scores after that many mini-batches"
    )
    parser.add_argument(
        "--hyperparams",
        type=create_dict,
        default="",
        help=("Define extra hyper parameters that will be used by samplers, "
              "models, wrappers and the training procedure. Some of them are "
              "lr, lr_reductions, seed, batch_size")
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save the actual and predicted scores"
    )
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="Save the validation detailed scores instead of just the average"
    )
    parser.add_argument(
        "--save_train_scores",
        action="store_true",
        help="Save the training scores instead of just the test scores"
    )
    parser.add_argument(
        "--plot_model",
        action="store_true",
        help="Plot the model being used"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the model weights"
    )
    parser.add_argument(
        "--snapshot_period",
        type=int,
        default=3000,
        help="The snapshot period in minibatches"
    )
    parser.add_argument(
        "--initial_weights",
        help="Initialize the model to those weights"
    )

    args = parser.parse_args(argv)

    # Seed the PRNG
    np.random.seed(args.hyperparams.get("seed", 0))
    tf_config.set_random_seed(np.random.randint(2**31-1))

    # Load the data, the model and the sampler
    dataset = load_dataset(args.dataset, args.hyperparams)
    reweighting_policy = build_reweighting_policy(
        args.reweighting,
        args.hyperparams
    )
    model = build_model(
        args.model,
        args.wrapper,
        dataset,
        args.hyperparams,
        reweighting_policy
    )
    sampler = build_sampler(
        args.sampler,
        dataset,
        model,
        args.hyperparams,
        reweighting_policy
    )

    # Plot the model if requested and available
    if args.plot_model:
        try:
            model.model.summary()
            plot_model(
                model=model.model,
                show_shapes=True,
                show_layer_names=False,
                to_file=path.join(args.output_directory, "model.png")
            )
        except AttributeError:
            pass

    # Connect output and logging functions to relevant signals
    training_progress = metrics_progress(args.output_directory, "train.txt")
    evaluation_progress = EvaluationLogger(
        args.output_directory,
        "val_eval.txt",
        "val_scores.txt",
        "train_eval.txt",
        "train_scores.txt"
    )
    comparison_logger = log_lines(
        args.output_directory,
        "comparison.txt",
        multiple=True
    )
    prediction_logger = PredictionLogger(
        args.output_directory,
        "predictions.txt"
    )
    signal("is.training").connect(training_progress)
    signal("is.evaluation").connect(evaluation_progress.on_evaluate)
    signal("is.lstm_comparison_sampler.scores").connect(comparison_logger)
    if args.save_predictions:
        signal("is.sample").connect(prediction_logger.on_sample)
        signal("is.training").connect(prediction_logger.on_training)
    if args.save_scores:
        signal("is.evaluate_batch").connect(evaluation_progress.on_batch)

    # Initialize the weights if needed
    if args.initial_weights:
        model.model.load_weights(args.initial_weights)

    # Start training
    should_validate = every_nth(args.validate_every)
    should_snapshot = every_nth(args.snapshot_period)
    clock = WallClock(path.join(args.output_directory, "clock.txt"))
    # Main training loop
    time_based = args.hyperparams.get("time_based", 0)
    lr = args.hyperparams.get("lr", 1e-3)
    lr_changes = args.hyperparams.get("lr_changes", [])
    lr_targets = [lr] + args.hyperparams.get("lr_targets", [])
    current_lr = None
    batch_size = args.hyperparams.get("batch_size", 128)
    train_idxs_step = max(1, len(dataset.train_data) // len(dataset.test_data))
    train_idxs = np.arange(len(dataset.train_data))[::train_idxs_step]
    b = 0
    i = 0
    while b < args.train_for:
        # Set the learning rate for this mini batch
        new_lr = lr_targets[bisect_right(lr_changes, b)]
        if new_lr != current_lr:
            model.set_lr(new_lr)
            current_lr = new_lr
            sys.stderr.write("Setting lr: %f\n" % (new_lr,))
            sys.stderr.flush()
        # Sample some points with their respective weights
        idxs, (x, y), w = sampler.sample(batch_size)
        # Train on the sampled points
        losses, metrics, scores = model.train_batch(x, y, w)
        # Update the sampler
        sampler.update(idxs, scores)
        # Write down the elapsed time
        clock.write()
        # Compute the validation score if we have to
        if should_validate():
            evaluation_progress.save_test()
            model.evaluate(*dataset.test_data[:])
            if args.save_train_scores:
                evaluation_progress.save_train()
                model.evaluate(*dataset.train_data[train_idxs])
        # Save the model weights
        if args.save_model and should_snapshot():
            model.model.save_weights(
                path.join(args.output_directory, "model.%06d.h5" % (i,))
            )
            with ignored(AttributeError):
                model.small_model.save_weights(
                    path.join(args.output_directory, "small_model.%06d.h5" % (i,))
                )
            with ignored(AttributeError):
                sampler.model.save_weights(
                    path.join(args.output_directory, "sampler_model.%06d.h5" % (i,))
                )
        # Update the iterations variable b and i
        i += 1
        if time_based:
            b = clock.elapsed
        else:
            b += 1

    if args.save_model:
        model.model.save_weights(
            path.join(args.output_directory, "model.h5")
        )
        with ignored(AttributeError):
            model.small_model.save_weights(
                path.join(args.output_directory, "small_model.h5" % (i,))
            )
        with ignored(AttributeError):
            sampler.model.save_weights(
                path.join(args.output_directory, "sampler_model.h5" % (i,))
            )


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
