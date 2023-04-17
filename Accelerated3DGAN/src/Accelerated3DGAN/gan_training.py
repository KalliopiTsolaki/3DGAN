## Import
from __future__ import print_function
import os

import glob

from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle

import argparse
import sys
#import h5py
import numpy as np
import time
import math
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
from tensorflow.keras.utils import Progbar

from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Lambda,
    Dropout,
    Activation,
    Embedding,
)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import (
    UpSampling3D,
    Conv3D,
    ZeroPadding3D,
    AveragePooling3D,
)
from tensorflow.keras.models import Model, Sequential
import math

import json


## Loss Calculation


def compute_global_loss(
    labels, predictions, global_batch_size, loss_weights=[3, 0.1, 25, 0.1]
):
    """_summary_

    Args:
        labels (_type_): _description_
        predictions (_type_): _description_
        global_batch_size (_type_): _description_
        loss_weights (list, optional): _description_. Defaults to [3, 0.1, 25, 0.1].

    Returns:
        _type_: _description_
    """

    # can be initialized outside
    binary_crossentropy_object = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
    mean_absolute_percentage_error_object = tf.keras.losses.MeanAbsolutePercentageError(
        reduction=tf.keras.losses.Reduction.NONE
    )
    mae_object = tf.keras.losses.MeanAbsoluteError(
        reduction=tf.keras.losses.Reduction.NONE
    )

    binary_example_loss = binary_crossentropy_object(
        labels[0], predictions[0], sample_weight=loss_weights[0]
    )
    mean_example_loss_1 = mean_absolute_percentage_error_object(
        labels[1], predictions[1], sample_weight=loss_weights[1]
    )
    mae_example_loss = mae_object(
        labels[2], predictions[2], sample_weight=loss_weights[2]
    )
    mean_example_loss_2 = mean_absolute_percentage_error_object(
        labels[3], predictions[3], sample_weight=loss_weights[3]
    )

    binary_loss = tf.nn.compute_average_loss(
        binary_example_loss, global_batch_size=global_batch_size
    )  # , sample_weight=1/loss_weights[0])
    mean_loss_1 = tf.nn.compute_average_loss(
        mean_example_loss_1, global_batch_size=global_batch_size
    )  # , sample_weight=1/loss_weights[1])
    mae_loss = tf.nn.compute_average_loss(
        mae_example_loss, global_batch_size=global_batch_size
    )  # , sample_weight=1/loss_weights[2])
    mean_loss_2 = tf.nn.compute_average_loss(
        mean_example_loss_2, global_batch_size=global_batch_size
    )  # , sample_weight=1/loss_weights[3])

    return [binary_loss, mean_loss_1, mae_loss, mean_loss_2]


# auxiliar functions


def hist_count(x, p=1.0, daxis=(1, 2, 3)):
    """_summary_

    Args:
        x (array): _description_
        p (float, optional): _description_. Defaults to 1.0.
        daxis (tuple, optional): _description_. Defaults to (1, 2, 3).

    Returns:
        _type_: _description_
    """
    limits = np.array([0.05, 0.03, 0.02, 0.0125, 0.008, 0.003])  # bin boundaries used
    limits = np.power(limits, p)
    bin1 = np.sum(np.where(x > (limits[0]), 1, 0), axis=daxis)
    bin2 = np.sum(np.where((x < (limits[0])) & (x > (limits[1])), 1, 0), axis=daxis)
    bin3 = np.sum(np.where((x < (limits[1])) & (x > (limits[2])), 1, 0), axis=daxis)
    bin4 = np.sum(np.where((x < (limits[2])) & (x > (limits[3])), 1, 0), axis=daxis)
    bin5 = np.sum(np.where((x < (limits[3])) & (x > (limits[4])), 1, 0), axis=daxis)
    bin6 = np.sum(np.where((x < (limits[4])) & (x > (limits[5])), 1, 0), axis=daxis)
    bin7 = np.sum(np.where((x < (limits[5])) & (x > 0.0), 1, 0), axis=daxis)
    bin8 = np.sum(np.where(x == 0, 1, 0), axis=daxis)
    bins = np.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1)
    bins[
        np.where(bins == 0)
    ] = 1  # so that an empty bin will be assigned a count of 1 to avoid unstability
    return bins


def BitFlip(x, prob=0.05):
    """
    Flips a single bit according to a certain probability.

    Args:
        x (list): list of bits to be flipped
        prob (float): probability of flipping one bit

    Returns:
        list: List of flipped bits

    """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


# Training

#@tf.function(jit_compile=True)
tf.function()
def Train_steps(
    dataset,
    generator,
    discriminator,
    latent_size,
    batch_size_per_replica,
    batch_size,
    loss_weights,
    optimizer_discriminator,
    optimizer_generator,
):
    """_summary_

    Args:
        dataset (_type_): _description_
        generator (_type_): _description_
        discriminator (_type_): _description_
        latent_size (_type_): _description_
        batch_size_per_replica (_type_): _description_
        batch_size (_type_): _description_
        loss_weights (_type_): _description_
        optimizer_discriminator (_type_): _description_
        optimizer_generator (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Get a single batch
    image_batch = dataset.get("X")  # .numpy()
    energy_batch = dataset.get("Y")  # .numpy()
    ecal_batch = dataset.get("ecal")  # .numpy()
    ang_batch = dataset.get("ang")  # .numpy()
    # add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

    # Generate Fake events with same energy and angle as data batch
    noise = tf.random.normal((batch_size_per_replica, latent_size - 2), 0, 1)
    generator_ip = tf.concat(
        (tf.reshape(energy_batch, (-1, 1)), tf.reshape(ang_batch, (-1, 1)), noise),
        axis=1,
    )
    generated_images = generator(generator_ip, training=False)

    # Train discriminator first on real batch
    fake_batch = BitFlip(np.ones(batch_size_per_replica).astype(np.float32))
    fake_batch = [[el] for el in fake_batch]
    labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

    with tf.GradientTape() as tape:
        predictions = discriminator(image_batch, training=True)
        real_batch_loss = compute_global_loss(
            labels, predictions, batch_size, loss_weights=loss_weights
        )

    gradients = tape.gradient(
        real_batch_loss, discriminator.trainable_variables
    )  # model.trainable_variables or  model.trainable_weights

    # ------------Minimize------------
    # aggregate_grads_outside_optimizer = (optimizer._HAS_AGGREGATE_GRAD and not isinstance(strategy.extended, parameter_server_strategy.))
    #gradients = optimizer_discriminator._clip_gradients(gradients)

    # --------------------------------

    optimizer_discriminator.apply_gradients(
        zip(gradients, discriminator.trainable_variables)
    )  # model.trainable_variables or  model.trainable_weights

    # Train discriminato on the fake batch
    fake_batch = BitFlip(np.zeros(batch_size_per_replica).astype(np.float32))
    fake_batch = [[el] for el in fake_batch]
    labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

    with tf.GradientTape() as tape:
        predictions = discriminator(generated_images, training=True)
        fake_batch_loss = compute_global_loss(
            labels, predictions, batch_size, loss_weights=loss_weights
        )
    gradients = tape.gradient(
        fake_batch_loss, discriminator.trainable_variables
    )  # model.trainable_variables or  model.trainable_weights
    #gradients = optimizer_discriminator._clip_gradients(gradients)
    optimizer_discriminator.apply_gradients(
        zip(gradients, discriminator.trainable_variables)
    )  # model.trainable_variables or  model.trainable_weights

    trick = np.ones(batch_size_per_replica).astype(np.float32)
    fake_batch = [[el] for el in trick]
    labels = [fake_batch, tf.reshape(energy_batch, (-1, 1)), ang_batch, ecal_batch]

    gen_losses = []
    # Train generator twice using combined model
    for _ in range(2):
        noise = tf.random.normal((batch_size_per_replica, latent_size - 2), 0, 1)
        generator_ip = tf.concat(
            (tf.reshape(energy_batch, (-1, 1)), tf.reshape(ang_batch, (-1, 1)), noise),
            axis=1,
        )  # sampled angle same as g4 theta

        with tf.GradientTape() as tape:
            generated_images = generator(generator_ip, training=True)
            predictions = discriminator(generated_images, training=True)
            loss = compute_global_loss(
                labels, predictions, batch_size, loss_weights=loss_weights
            )

        gradients = tape.gradient(
            loss, generator.trainable_variables
        )  # model.trainable_variables or  model.trainable_weights
        #gradients = optimizer_generator._clip_gradients(gradients)
        optimizer_generator.apply_gradients(
            zip(gradients, generator.trainable_variables)
        )  # model.trainable_variables or  model.trainable_weights

        for el in loss:
            gen_losses.append(el)

    return (
        real_batch_loss[0],
        real_batch_loss[1],
        real_batch_loss[2],
        real_batch_loss[3],
        fake_batch_loss[0],
        fake_batch_loss[1],
        fake_batch_loss[2],
        fake_batch_loss[3],
        gen_losses[0],
        gen_losses[1],
        gen_losses[2],
        gen_losses[3],
        gen_losses[4],
        gen_losses[5],
        gen_losses[6],
        gen_losses[7],
    )


def Test_steps(
    dataset,
    generator,
    discriminator,
    latent_size,
    batch_size_per_replica,
    batch_size,
    loss_weights,
):
    # Get a single batch
    image_batch = dataset.get("X")  # .numpy()
    energy_batch = dataset.get("Y")  # .numpy()
    ecal_batch = dataset.get("ecal")  # .numpy()
    ang_batch = dataset.get("ang")  # .numpy()
    # add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

    # Generate Fake events with same energy and angle as data batch
    noise = np.random.normal(0, 1, (batch_size_per_replica, latent_size - 2)).astype(
        np.float32
    )
    generator_ip = tf.concat(
        (tf.reshape(energy_batch, (-1, 1)), tf.reshape(ang_batch, (-1, 1)), noise),
        axis=1,
    )
    print(generator_ip.shape)
    generated_images = generator(generator_ip, training=False)

    # concatenate to fake and real batches
    X = tf.concat((image_batch, generated_images), axis=0)
    y = np.array([1] * batch_size_per_replica + [0] * batch_size_per_replica).astype(
        np.float32
    )
    ang = tf.concat((ang_batch, ang_batch), axis=0)
    ecal = tf.concat((ecal_batch, ecal_batch), axis=0)
    aux_y = tf.concat((energy_batch, energy_batch), axis=0)
    # add_loss= tf.concat((add_loss_batch, add_loss_batch), axis=0)

    y = [[el] for el in y]

    labels = [y, aux_y, ang, ecal]
    disc_eval = discriminator(X, training=False)
    disc_eval_loss = compute_global_loss(
        labels, disc_eval, batch_size, loss_weights=loss_weights
    )

    trick = np.ones(batch_size_per_replica).astype(
        np.float32
    )  # original doest have astype
    fake_batch = [[el] for el in trick]
    labels = [fake_batch, energy_batch, ang_batch, ecal_batch]
    generated_images = generator(generator_ip, training=False)
    gen_eval = discriminator(
        generated_images, training=False
    )  # combined(generator_ip, training=False)

    gen_eval_loss = compute_global_loss(
        labels, gen_eval, batch_size, loss_weights=loss_weights
    )

    return (
        disc_eval_loss[0],
        disc_eval_loss[1],
        disc_eval_loss[2],
        disc_eval_loss[3],
        gen_eval_loss[0],
        gen_eval_loss[1],
        gen_eval_loss[2],
        gen_eval_loss[3],
    )


@tf.function()
def distributed_train_step(
    strategy,
    dataset,
    generator,
    discriminator,
    latent_size,
    batch_size_per_replica,
    batch_size,
    loss_weights,
    optimizer_discriminator,
    optimizer_generator,
):

    gen_losses = []
    (
        real_batch_loss_1,
        real_batch_loss_2,
        real_batch_loss_3,
        real_batch_loss_4,
        fake_batch_loss_1,
        fake_batch_loss_2,
        fake_batch_loss_3,
        fake_batch_loss_4,
        gen_batch_loss_1,
        gen_batch_loss_2,
        gen_batch_loss_3,
        gen_batch_loss_4,
        gen_batch_loss_5,
        gen_batch_loss_6,
        gen_batch_loss_7,
        gen_batch_loss_8,
    ) = strategy.run(
        Train_steps,
        args=(
            next(dataset),
            generator,
            discriminator,
            latent_size,
            batch_size_per_replica,
            batch_size,
            loss_weights,
            optimizer_discriminator,
            optimizer_generator,
        ),
    )

    real_batch_loss_1 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, real_batch_loss_1, axis=None
    )
    real_batch_loss_2 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, real_batch_loss_2, axis=None
    )
    real_batch_loss_3 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, real_batch_loss_3, axis=None
    )
    real_batch_loss_4 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, real_batch_loss_4, axis=None
    )
    real_batch_loss = [
        real_batch_loss_1,
        real_batch_loss_2,
        real_batch_loss_3,
        real_batch_loss_4,
    ]

    fake_batch_loss_1 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, fake_batch_loss_1, axis=None
    )
    fake_batch_loss_2 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, fake_batch_loss_2, axis=None
    )
    fake_batch_loss_3 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, fake_batch_loss_3, axis=None
    )
    fake_batch_loss_4 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, fake_batch_loss_4, axis=None
    )
    fake_batch_loss = [
        fake_batch_loss_1,
        fake_batch_loss_2,
        fake_batch_loss_3,
        fake_batch_loss_4,
    ]

    gen_batch_loss_1 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_batch_loss_1, axis=None
    )
    gen_batch_loss_2 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_batch_loss_2, axis=None
    )
    gen_batch_loss_3 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_batch_loss_3, axis=None
    )
    gen_batch_loss_4 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_batch_loss_4, axis=None
    )
    gen_batch_loss = [
        gen_batch_loss_1,
        gen_batch_loss_2,
        gen_batch_loss_3,
        gen_batch_loss_4,
    ]

    gen_losses.append(gen_batch_loss)

    gen_batch_loss_5 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_batch_loss_5, axis=None
    )
    gen_batch_loss_6 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_batch_loss_6, axis=None
    )
    gen_batch_loss_7 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_batch_loss_7, axis=None
    )
    gen_batch_loss_8 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_batch_loss_8, axis=None
    )
    gen_batch_loss = [
        gen_batch_loss_5,
        gen_batch_loss_6,
        gen_batch_loss_7,
        gen_batch_loss_8,
    ]

    gen_losses.append(gen_batch_loss)

    return real_batch_loss, fake_batch_loss, gen_losses


@tf.function()
def distributed_test_step(
    strategy,
    dataset,
    generator,
    discriminator,
    latent_size,
    batch_size_per_replica,
    batch_size,
    loss_weights,
):
    (
        disc_test_loss_1,
        disc_test_loss_2,
        disc_test_loss_3,
        disc_test_loss_4,
        gen_test_loss_1,
        gen_test_loss_2,
        gen_test_loss_3,
        gen_test_loss_4,
    ) = strategy.run(
        Test_steps,
        args=(
            next(dataset),
            generator,
            discriminator,
            latent_size,
            batch_size_per_replica,
            batch_size,
            loss_weights,
        ),
    )

    disc_test_loss_1 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, disc_test_loss_1, axis=None
    )
    disc_test_loss_2 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, disc_test_loss_2, axis=None
    )
    disc_test_loss_3 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, disc_test_loss_3, axis=None
    )
    disc_test_loss_4 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, disc_test_loss_4, axis=None
    )
    disc_test_loss = [
        disc_test_loss_1,
        disc_test_loss_2,
        disc_test_loss_3,
        disc_test_loss_4,
    ]

    gen_test_loss_1 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_test_loss_1, axis=None
    )
    gen_test_loss_2 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_test_loss_2, axis=None
    )
    gen_test_loss_3 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_test_loss_3, axis=None
    )
    gen_test_loss_4 = strategy.reduce(
        tf.distribute.ReduceOp.SUM, gen_test_loss_4, axis=None
    )
    gen_test_loss = [gen_test_loss_1, gen_test_loss_2, gen_test_loss_3, gen_test_loss_4]

    return disc_test_loss, gen_test_loss
