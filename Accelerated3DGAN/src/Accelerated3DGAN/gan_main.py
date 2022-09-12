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
import h5py
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

## import files
from gan_dataset import *
from gan_training import *
from gan_models import *

# Configs
def get_parser():
    """Parser with the arguments possible when runing the program

    Returns:
        Parser: The parser object
    """
    parser = argparse.ArgumentParser(description="3D GAN Params")
    parser.add_argument("--multi_node", action="store", default=False)
    parser.add_argument(
        "--workers", nargs="+", default=""
    )  # use like this --workers 10.1.10.58:12345 10.1.10.250:12345
    parser.add_argument("--index", action="store", type=int, default=0)
    parser.add_argument("--use_gs", action="store", default=False)
    # parser.add_argument('--datapath', action='store', default='')
    # parser.add_argument('--outpath', action='store', default='')
    parser.add_argument(
        "--datapath",
        action="store",
        default="/eos/user/r/redacost/tfrecordsprepro/*.tfrecords",
        help="Data path",
    )
    parser.add_argument(
        "--outpath",
        action="store",
        default="/eos/user/r/redacost/tfresults/",
        help="training output",
    )
    parser.add_argument(
        "--nbepochs",
        action="store",
        type=int,
        default=60,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--batchsize",
        action="store",
        type=int,
        default=64,
        help="batch size per update",
    )
    parser.add_argument(
        "--use_gpus", action="store", default=True, help="Use gpus for training"
    )
    parser.add_argument("--GLOBAL_BATCH_SIZE", action="store", default=64)
    parser.add_argument("--nb_epochs", action="store", default=60, help="Total Epochs")
    parser.add_argument("--batch_size", action="store", default=64)
    parser.add_argument(
        "--latent_size", action="store", default=256, help="latent vector size"
    )
    parser.add_argument("--verbose", action="store", default=True)
    parser.add_argument(
        "--nEvents",
        action="store",
        default=400000,
        help="maximum number of events used in training",
    )
    parser.add_argument("--ascale", action="store", default=1, help="angle scale")
    parser.add_argument("--yscale", action="store", default=100, help="scaling energy")
    parser.add_argument("--xscale", action="store", default=1)
    parser.add_argument("--xpower", action="store", default=0.85)
    parser.add_argument("--angscale", action="store", default=1)
    parser.add_argument("--analyse", action="store", default=False, help="if analysing")
    parser.add_argument("--dformat", action="store", default="channels_first")
    parser.add_argument(
        "--thresh", action="store", default=0, help="threshold for data"
    )
    parser.add_argument("--angtype", action="store", default="mtheta")
    parser.add_argument("--particle", action="store", default="Ele")
    parser.add_argument("--warm", action="store", default=False)
    parser.add_argument("--lr", action="store", default=0.001)
    parser.add_argument("--events_per_file", action="store", default=5000)
    parser.add_argument("--name", action="store", default="gan_training")

    parser.add_argument(
        "--g_weights", action="store", default="params_generator_epoch_"
    )
    parser.add_argument(
        "--d_weights", action="store", default="params_discriminator_epoch_"
    )

    parser.add_argument("--tlab", action="store", default=False)
    return parser


def main_gan():
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    main_session = tf.compat.v1.InteractiveSession(config=config)

    parser = get_parser()
    params = parser.parse_args()

    GLOBAL_BATCH_SIZE = params.GLOBAL_BATCH_SIZE
    nb_epochs = params.nb_epochs
    batch_size = params.batch_size
    latent_size = params.latent_size
    verbose = params.verbose
    nEvents = params.nEvents
    ascale = params.ascale
    yscale = params.yscale
    xscale = params.xscale
    xpower = params.xpower
    angscale = params.angscale
    analyse = params.analyse
    dformat = params.dformat
    thresh = params.thresh
    angtype = params.angtype
    particle = params.particle
    warm = params.warm
    lr = params.lr
    events_per_file = params.events_per_file
    name = params.name

    g_weights = params.g_weights
    d_weights = params.d_weights

    tlab = params.tlab

    multi_node = params.multi_node
    use_gs = params.use_gs
    use_gpus = params.use_gpus

    if not params.datapath == "":
        datapath = params.datapath
    if not params.outpath == "":
        outpath = params.outpath
    if not params.nbepochs == "":
        nb_epochs = params.nbepochs
    if not params.batchsize == "":
        batch_size = params.batchsize

    if use_gpus:
        if multi_node:
            workers = params.workers
            index = params.index

            print(multi_node)
            print(workers)
            print(index)

            # tf_config
            os.environ["TF_CONFIG"] = json.dumps(
                {
                    "cluster": {
                        "worker": workers
                    },  # ["10.1.10.58:12345", "10.1.10.250:12345"]},
                    "task": {"type": "worker", "index": index},
                }
            )

    ## Initialization
    WeightsDir = outpath + "weights/3dgan_weights_" + name
    pklfile = outpath + "results/3dgan_history_" + name + ".pkl"  # loss history
    resultfile = (
        outpath + "results/3dgan_analysis" + name + ".pkl"
    )  # optimization metric history
    prev_gweights = ""  # outpath + 'weights/' + params.prev_gweights
    prev_dweights = ""  # outpath + 'weights/' + params.prev_dweights

    # loss_weights=[params.gen_weight, params.aux_weight, params.ang_weight, params.ecal_weight]
    loss_weights = [3, 0.1, 25, 0.1]
    energies = [0, 110, 150, 190, 1]

    # Define Strategy and models
    if use_gpus:
        if not multi_node:
            strategy = tf.distribute.MirroredStrategy()
        if multi_node:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
                tf.distribute.experimental.CollectiveCommunication.NCCL
            )
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        BATCH_SIZE_PER_REPLICA = batch_size
        batch_size = batch_size * strategy.num_replicas_in_sync
        batch_size_per_replica = BATCH_SIZE_PER_REPLICA

    else:
        batch_size_per_replica = batch_size

    # Compilation of models and definition of train/test files

    start_init = time.time()
    f = [0.9, 0.1]  # train, test fractions

    # loss_ftn = hist_count # function used for additional loss

    # apply settings according to data format
    if dformat == "channels_last":
        daxis = 4  # channel axis
        daxis2 = (1, 2, 3)  # axis for sum
    else:
        daxis = 1  # channel axis
        daxis2 = (2, 3, 4)  # axis for sum

    # Trainfiles, Testfiles = DivideFiles(datapath, f, datasetnames=["ECAL"], Particles =[particle])
    if not use_gs:
        Trainfiles, Testfiles = DivideFiles(
            datapath, f, datasetnames=["ECAL"], Particles=[particle]
        )
    if use_gs:
        Trainfiles = [
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_000.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_001.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_002.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_003.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_004.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_005.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_006.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_007.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_008.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_009.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_010.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_011.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_012.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_013.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_014.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_015.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_016.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_017.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_018.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_019.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_020.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_021.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_022.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_023.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_024.tfrecords",
        ]
        Testfiles = [
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_025.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_026.tfrecords",
            "gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_027.tfrecords",
        ]

    print(Trainfiles)
    print(Testfiles)

    nb_Test = int(
        nEvents * f[1]
    )  # The number of test events calculated from fraction of nEvents
    nb_Train = int(
        nEvents * f[0]
    )  # The number of train events calculated from fraction of nEvents

    # create history and finish initiation
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    init_time = time.time() - start_init
    analysis_history = defaultdict(list)
    print("Initialization time is {} seconds".format(init_time))

    if use_gpus:
        with strategy.scope():
            discriminator = discriminator_model(xpower, dformat=dformat)
            generator = generator_model(latent_size, dformat=dformat)
            optimizer_discriminator = RMSprop(lr)
            optimizer_generator = RMSprop(lr)
    else:
        discriminator = discriminator_model(xpower, dformat=dformat)
        generator = generator_model(latent_size, dformat=dformat)
        optimizer_discriminator = RMSprop(lr)
        optimizer_generator = RMSprop(lr)

    print("Loading Data")

    if use_gpus:
        dataset = RetrieveTFRecordpreprocessing(Trainfiles, batch_size)
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        dist_dataset_iter = iter(dist_dataset)

        test_dataset = RetrieveTFRecordpreprocessing(Testfiles, batch_size)
        test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
        test_dist_dataset_iter = iter(test_dist_dataset)
    else:
        dataset = RetrieveTFRecordpreprocessing(Trainfiles, batch_size)
        dist_dataset_iter = iter(dataset)

        test_dataset = RetrieveTFRecordpreprocessing(Testfiles, batch_size)
        test_dist_dataset_iter = iter(test_dataset)

    # needs to change so it is not hard coded
    # steps_per_epoch =int( datasetsize // (batch_size))
    steps_per_epoch = int(124987 // (batch_size))
    # test_steps_per_epoch =int( datasetsizetest // (batch_size))
    test_steps_per_epoch = int(12340 // (batch_size))

    # Start training
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        print("Epoch {} of {}".format(epoch + 1, nb_epochs))

        # --------------------------------------------------------------------------------------------
        # ------------------------------ Main Training Cycle -----------------------------------------
        # --------------------------------------------------------------------------------------------

        # Get the data for each training file

        nb_file = 0
        epoch_gen_loss = []
        epoch_disc_loss = []
        index = 0
        file_index = 0
        nbatch = 0

        print("Number of Batches: ", steps_per_epoch)

        for _ in range(steps_per_epoch):
            file_time = time.time()

            # Discriminator Training
            if use_gpus:
                real_batch_loss, fake_batch_loss, gen_losses = distributed_train_step(
                    strategy,
                    dist_dataset_iter,
                    generator,
                    discriminator,
                    latent_size,
                    batch_size_per_replica,
                    batch_size,
                    loss_weights,
                    optimizer_discriminator,
                    optimizer_generator,
                )
            else:
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
                ) = Train_steps(
                    next(dist_dataset_iter),
                    generator,
                    discriminator,
                    latent_size,
                    batch_size_per_replica,
                    batch_size,
                    loss_weights,
                    optimizer_discriminator,
                    optimizer_generator,
                )
                real_batch_loss = [
                    real_batch_loss_1,
                    real_batch_loss_2,
                    real_batch_loss_3,
                    real_batch_loss_4,
                ]
                fake_batch_loss = [
                    fake_batch_loss_1,
                    fake_batch_loss_2,
                    fake_batch_loss_3,
                    fake_batch_loss_4,
                ]
                gen_batch_loss = [
                    gen_batch_loss_1,
                    gen_batch_loss_2,
                    gen_batch_loss_3,
                    gen_batch_loss_4,
                ]
                gen_losses.append(gen_batch_loss)
                gen_batch_loss = [
                    gen_batch_loss_5,
                    gen_batch_loss_6,
                    gen_batch_loss_7,
                    gen_batch_loss_8,
                ]
                gen_losses.append(gen_batch_loss)

            # Configure the loss so it is equal to the original values
            real_batch_loss = [el.numpy() for el in real_batch_loss]
            real_batch_loss_total_loss = np.sum(real_batch_loss)
            new_real_batch_loss = [real_batch_loss_total_loss]
            for i_weights in range(len(real_batch_loss)):
                new_real_batch_loss.append(
                    real_batch_loss[i_weights] / loss_weights[i_weights]
                )
            real_batch_loss = new_real_batch_loss

            fake_batch_loss = [el.numpy() for el in fake_batch_loss]
            fake_batch_loss_total_loss = np.sum(fake_batch_loss)
            new_fake_batch_loss = [fake_batch_loss_total_loss]
            for i_weights in range(len(fake_batch_loss)):
                new_fake_batch_loss.append(
                    fake_batch_loss[i_weights] / loss_weights[i_weights]
                )
            fake_batch_loss = new_fake_batch_loss

            # if ecal sum has 100% loss(generating empty events) then end the training
            if fake_batch_loss[3] == 100.0 and index > 10:
                print(
                    "Empty image with Ecal loss equal to 100.0 for {} batch".format(
                        index
                    )
                )
                generator.save_weights(
                    WeightsDir + "/{0}eee.hdf5".format(g_weights), overwrite=True
                )
                discriminator.save_weights(
                    WeightsDir + "/{0}eee.hdf5".format(d_weights), overwrite=True
                )
                print("real_batch_loss", real_batch_loss)
                print("fake_batch_loss", fake_batch_loss)
                sys.exit()

            # append mean of discriminator loss for real and fake events
            epoch_disc_loss.append(
                [(a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)]
            )

            gen_losses[0] = [el.numpy() for el in gen_losses[0]]
            gen_losses_total_loss = np.sum(gen_losses[0])
            new_gen_losses = [gen_losses_total_loss]
            for i_weights in range(len(gen_losses[0])):
                new_gen_losses.append(
                    gen_losses[0][i_weights] / loss_weights[i_weights]
                )
            gen_losses[0] = new_gen_losses

            gen_losses[1] = [el.numpy() for el in gen_losses[1]]
            gen_losses_total_loss = np.sum(gen_losses[1])
            new_gen_losses = [gen_losses_total_loss]
            for i_weights in range(len(gen_losses[1])):
                new_gen_losses.append(
                    gen_losses[1][i_weights] / loss_weights[i_weights]
                )
            gen_losses[1] = new_gen_losses

            generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]

            epoch_gen_loss.append(generator_loss)

            print(
                "Time taken by batch",
                str(nbatch),
                " was",
                str(time.time() - file_time),
                "seconds.",
            )
            nbatch += 1

        print(
            "Time taken by epoch{} was {} seconds.".format(
                epoch, time.time() - epoch_start
            )
        )
        train_time = time.time() - epoch_start

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # --------------------------------------------------------------------------------------------
        # ------------------------------ Main Testing Cycle ------------------------------------------
        # --------------------------------------------------------------------------------------------

        # read first test file
        disc_test_loss = []
        gen_test_loss = []
        nb_file = 0
        index = 0
        file_index = 0

        # Test process will also be accomplished in batches to reduce memory consumption
        print("\nTesting for epoch {}:".format(epoch))
        test_start = time.time()

        # Testing
        # add Testfiles, nb_test_batches, daxis, daxis2, X_train(??), loss_ftn, combined
        for _ in range(test_steps_per_epoch):

            this_batch_size = 128  # can be removed (should)

            if use_gpus:
                disc_eval_loss, gen_eval_loss = distributed_test_step(
                    strategy,
                    test_dist_dataset_iter,
                    generator,
                    discriminator,
                    latent_size,
                    batch_size_per_replica,
                    batch_size,
                    loss_weights,
                )
            else:
                (
                    disc_test_loss_1,
                    disc_test_loss_2,
                    disc_test_loss_3,
                    disc_test_loss_4,
                    gen_test_loss_1,
                    gen_test_loss_2,
                    gen_test_loss_3,
                    gen_test_loss_4,
                ) = Test_steps(
                    next(test_dist_dataset_iter),
                    generator,
                    discriminator,
                    latent_size,
                    batch_size_per_replica,
                    batch_size,
                    loss_weights,
                )

                disc_test_loss = [
                    disc_test_loss_1,
                    disc_test_loss_2,
                    disc_test_loss_3,
                    disc_test_loss_4,
                ]
                gen_test_loss = [
                    gen_test_loss_1,
                    gen_test_loss_2,
                    gen_test_loss_3,
                    gen_test_loss_4,
                ]

            # Configure the loss so it is equal to the original values
            disc_eval_loss = [el.numpy() for el in disc_eval_loss]
            disc_eval_loss_total_loss = np.sum(disc_eval_loss)
            new_disc_eval_loss = [disc_eval_loss_total_loss]
            for i_weights in range(len(disc_eval_loss)):
                new_disc_eval_loss.append(
                    disc_eval_loss[i_weights] / loss_weights[i_weights]
                )
            disc_eval_loss = new_disc_eval_loss

            gen_eval_loss = [el.numpy() for el in gen_eval_loss]
            gen_eval_loss_total_loss = np.sum(gen_eval_loss)
            new_gen_eval_loss = [gen_eval_loss_total_loss]
            for i_weights in range(len(gen_eval_loss)):
                new_gen_eval_loss.append(
                    gen_eval_loss[i_weights] / loss_weights[i_weights]
                )
            gen_eval_loss = new_gen_eval_loss

            index += 1
            # evaluate discriminator loss
            disc_test_loss.append(disc_eval_loss)
            # evaluate generator loss
            gen_test_loss.append(gen_eval_loss)

        # --------------------------------------------------------------------------------------------
        # ------------------------------ Updates -----------------------------------------------------
        # --------------------------------------------------------------------------------------------

        # make loss dict
        print("Total Test batches were {}".format(index))
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        discriminator_test_loss = np.mean(np.array(disc_test_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        generator_test_loss = np.mean(np.array(gen_test_loss), axis=0)
        train_history["generator"].append(generator_train_loss)
        train_history["discriminator"].append(discriminator_train_loss)
        test_history["generator"].append(generator_test_loss)
        test_history["discriminator"].append(discriminator_test_loss)
        # print losses
        # print('{0:<20s} | {1:6s} | {2:12s} | {3:12s}| {4:5s} | {5:8s}'.format(
        #     'component', *discriminator.metrics_names))
        print(discriminator.metrics_names)
        print("-" * 65)
        ROW_FMT = (
            "{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}"
        )
        print(ROW_FMT.format("generator (train)", *train_history["generator"][-1]))
        print(ROW_FMT.format("generator (test)", *test_history["generator"][-1]))
        print(
            ROW_FMT.format("discriminator (train)", *train_history["discriminator"][-1])
        )
        print(
            ROW_FMT.format("discriminator (test)", *test_history["discriminator"][-1])
        )

        # save weights every epoch
        generator.save_weights(
            WeightsDir + "/{0}{1:03d}.hdf5".format(g_weights, epoch), overwrite=True
        )
        discriminator.save_weights(
            WeightsDir + "/{0}{1:03d}.hdf5".format(d_weights, epoch), overwrite=True
        )

        epoch_time = time.time() - test_start
        print(
            "The Testing for {} epoch took {} seconds. Weights are saved in {}".format(
                epoch, epoch_time, WeightsDir
            )
        )

        # save loss dict to pkl file
        pickle.dump({"train": train_history, "test": test_history}, open(pklfile, "wb"))

        print("train-loss:" + str(train_history["generator"][-1][0]))

        # #--------------------------------------------------------------------------------------------
        # #------------------------------ Analysis ----------------------------------------------------
        # #--------------------------------------------------------------------------------------------

        # # if a short analysis is to be performed for each epoch
        # if analyse:
        #     print('analysing..........')
        #     atime = time.time()
        #     # load all test data
        #     for index, dtest in enumerate(Testfiles):
        #         if index == 0:
        #             X_test, Y_test, ang_test, ecal_test = GetDataAngle(dtest, xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
        #         else:
        #             if X_test.shape[0] < nb_Test:
        #                 X_temp, Y_temp, ang_temp,  ecal_temp = GetDataAngle(dtest, xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
        #                 X_test = np.concatenate((X_test, X_temp))
        #                 Y_test = np.concatenate((Y_test, Y_temp))
        #                 ang_test = np.concatenate((ang_test, ang_temp))
        #                 ecal_test = np.concatenate((ecal_test, ecal_temp))
        #     if X_test.shape[0] > nb_Test:
        #         X_test, Y_test, ang_test, ecal_test = X_test[:nb_Test], Y_test[:nb_Test], ang_test[:nb_Test], ecal_test[:nb_Test]
        #     else:
        #         nb_Test = X_test.shape[0] # the nb_test maybe different if total events are less than nEvents
        #     var=gan.sortEnergy([np.squeeze(X_test), Y_test, ang_test], ecal_test, energies, ang=1)
        #     result = gan.OptAnalysisAngle(var, generator, energies, xpower = xpower, concat=2)
        #     print('{} seconds taken by analysis'.format(time.time()-atime))
        #     analysis_history['total'].append(result[0])
        #     analysis_history['energy'].append(result[1])
        #     analysis_history['moment'].append(result[2])
        #     analysis_history['angle'].append(result[3])
        #     print('Result = ', result)
        #     # write analysis history to a pickel file
        #     pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))


if __name__ == "__main__":
    main_gan()
