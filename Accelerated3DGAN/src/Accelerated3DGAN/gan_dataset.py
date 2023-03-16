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

def get_parser():
    parser = argparse.ArgumentParser(description='TFRecord dataset Params' )
    parser.add_argument('--datapath', action='store', type=str, default='', help='HDF5 files to convert.')
    parser.add_argument('--outpath', action='store', type=str, default='', help='Dir to save the tfrecord files.')
    return parser

# Divide files in train and test lists
def DivideFiles(
    FileSearch="/data/LCD/*/*.h5",
    Fractions=[0.9, 0.1],
    datasetnames=["ECAL", "HCAL"],
    Particles=[],
    MaxFiles=-1,
):
    """Divide dataset files into two diferents fractions to be used for Train and Test

    Args:
        FileSearch (str, optional): Path to the dataset. Defaults to "/data/LCD/*/*.h5".
        Fractions (list, optional): probability of Train/Test. Defaults to [0.9, 0.1].
        datasetnames (list, optional): Not used. Defaults to ["ECAL", "HCAL"].
        Particles (list, optional): _description_. Defaults to [].
        MaxFiles (int, optional): Maximum number of files. Defaults to -1.

    Returns:
        list: List containing both lists of train and test
    """

    print("Searching in :", FileSearch)
    Files = sorted(glob.glob(FileSearch))
    print("Found {} files. ".format(len(Files)))
    FileCount = 0
    Samples = {}
    for F in Files:
        FileCount += 1
        basename = os.path.basename(F)
        ParticleName = basename.split("_")[0].replace("Escan", "")
        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName] = [(F)]
        if MaxFiles > 0:
            if FileCount > MaxFiles:
                break
    out = []
    for j in range(len(Fractions)):
        out.append([])
    SampleI = len(Samples.keys()) * [int(0)]
    for i, SampleName in enumerate(Samples):
        Sample = Samples[SampleName]
        NFiles = len(Sample)
        for j, Frac in enumerate(Fractions):
            EndI = int(SampleI[i] + round(NFiles * Frac))
            out[j] += Sample[SampleI[i] : EndI]
            SampleI[i] = EndI
    return out


def GetDataAngleParallel(
    dataset,
    xscale=1,
    xpower=1,
    yscale=100,
    angscale=1,
    angtype="theta",
    thresh=1e-4,
    daxis=-1,
):
    """Preprocess function for the dataset

    Args:
        dataset (str): Dataset file path
        xscale (int, optional): Value to scale the ECAL values. Defaults to 1.
        xpower (int, optional): Value to scale the ECAL values, exponentially. Defaults to 1.
        yscale (int, optional): Value to scale the energy values. Defaults to 100.
        angscale (int, optional): Value to scale the angle values. Defaults to 1.
        angtype (str, optional): Which type of angle to use. Defaults to "theta".
        thresh (_type_, optional): Maximum value for ECAL values. Defaults to 1e-4.
        daxis (int, optional): Axis to expand values. Defaults to -1.

    Returns:
       Dict: Dictionary containning the preprocessed dataset
    """
    X = np.array(dataset.get("ECAL")) * xscale
    Y = np.array(dataset.get("energy")) / yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    X = X[indexes]
    Y = Y[indexes]
    if angtype in dataset:
        ang = np.array(dataset.get(angtype))[indexes]
    # else:
    # ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal = ecal[indexes]
    ecal = np.expand_dims(ecal, axis=daxis)
    if xpower != 1.0:
        X = np.power(X, xpower)

    Y = [[el] for el in Y]
    ang = [[el] for el in ang]
    ecal = [[el] for el in ecal]

    final_dataset = {"X": X, "Y": Y, "ang": ang, "ecal": ecal}

    return final_dataset


def RetrieveTFRecord(recorddatapaths):
    """Retrieves the Tf Records without the preprocessing done
    It needs to return the dataset withthe necessary elements so the preprocessng can be done:
    ECAL
    ecalsize
    energy
    eta
    mtheta
    sum
    theta


    Args:
        recorddatapaths (string): Path to trecord directory

    Returns:
        dict: dataset
    """
    recorddata = tf.data.TFRecordDataset(recorddatapaths)

    # print(type(recorddata))

    retrieveddata = {
        "ECAL": tf.io.FixedLenSequenceFeature(
            (), dtype=tf.float32, allow_missing=True
        ),  # float32
        "ecalsize": tf.io.FixedLenSequenceFeature(
            (), dtype=tf.int64, allow_missing=True
        ),  # needs size of ecal so it can reconstruct the narray
        #'bins': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        "energy": tf.io.FixedLenSequenceFeature(
            (), dtype=tf.float32, allow_missing=True
        ),  # float32
        "eta": tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        "mtheta": tf.io.FixedLenSequenceFeature(
            (), dtype=tf.float32, allow_missing=True
        ),
        "sum": tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        "theta": tf.io.FixedLenSequenceFeature(
            (), dtype=tf.float32, allow_missing=True
        ),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, retrieveddata)

    parsed_dataset = recorddata.map(_parse_function)

    # return parsed_dataset

    # print(type(parsed_dataset))

    for parsed_record in parsed_dataset:
        dataset = parsed_record

    dataset["ECAL"] = tf.reshape(dataset["ECAL"], dataset["ecalsize"])

    dataset.pop("ecalsize")

    return dataset


def RetrieveTFRecordpreprocessing(recorddatapaths, batch_size):
    """Retrieves the Tf Records without the preprocessing done
        It has 4 elements:
        X: images of dimension 51x51x25
        Y:
        ang:
        ecal:
        It also batches the dataset

    Args:
        recorddatapaths (string): path to the Tf Records
        batch_size (int): integer with the batch size

    Returns:
        Tensorflow Dataset: Tensorflow type Dataset with the tf records
    """
    recorddata = tf.data.TFRecordDataset(
        recorddatapaths, num_parallel_reads=tf.data.experimental.AUTOTUNE
    )
    # recorddata = tf.data.TFRecordDataset(recorddatapaths)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    retrieveddata = {
        "X": tf.io.FixedLenSequenceFeature(
            (), dtype=tf.float32, allow_missing=True
        ),  # float32
        #'ecalsize': tf.io.FixedLenSequencFeature((), dtype=tf.int64, allow_missing=True), #needs size of ecal so it can reconstruct the narray
        "Y": tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),  # float32
        "ang": tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),
        "ecal": tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        data = tf.io.parse_single_example(example_proto, retrieveddata)
        data["X"] = tf.reshape(data["X"], [1, 51, 51, 25])
        # print(tf.shape(data['Y']))
        data["Y"] = tf.reshape(data["Y"], [1])
        data["ang"] = tf.reshape(data["ang"], [1])
        data["ecal"] = tf.reshape(data["ecal"], [1])
        # print(tf.shape(data['Y']))
        return data

    parsed_dataset = (
        recorddata.map(_parse_function)
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .with_options(options)
    )

    return parsed_dataset
    # return parsed_dataset, ds_size

#convert dataset with preprocessing
def ConvertH5toTFRecordPreprocessing(datafile,filenumber,datadirectory):
    # read file
    print('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    
    dataset = GetDataAngleParallel(f)

    dataset = tf.data.Dataset.from_tensor_slices((dataset.get('X'),dataset.get('Y'),dataset.get('ang'),dataset.get('ecal')))#.batch(128)

    tf.print(dataset)

    
    print('Start')
    def serialize(feature1,feature2,feature3,feature4):
        finaldata = tf.train.Example(
            features=tf.train.Features( 
                feature={
                    'X': convert_ECAL(feature1), #float32
                    #'ecalsize': convert_int_feature(list(dataset.get('X').shape)), #needs size of ecal so it can reconstruct the array
                    'Y': convert_floats(feature2, 'Y'), #float32
                    'ang': convert_floats(feature3, 'ang'), #float32
                    'ecal': convert_floats(feature4, 'ecal'), #float64
                }
            )
        )
        #seri += 1
        #print(seri)
        return finaldata.SerializeToString()

    def serialize_example(f0,f1,f2,f3):
        tf_string = tf.py_function(serialize,(f0,f1,f2,f3),tf.string)
        return tf.reshape(tf_string, ())


    serialized_dataset = dataset.map(serialize_example)
    print(serialized_dataset) 
        

    filename = datadirectory + '/Ele_VarAngleMeas_100_200_{0:03}.tfrecords'.format(filenumber)
    print('Writing data in .....', filename)
    writer = tf.data.experimental.TFRecordWriter(str(filename))
    writer.write(serialized_dataset)

    return serialized_dataset

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    datapath = params.datapath# Data path
    outpath = params.outpath # training output
    Files = sorted( glob.glob(datapath))
    print ("Found {} files. ".format(len(Files)))

    filenumber = 0

    for f in Files:
        print(filenumber)
        finaldata = ConvertH5toTFRecordPreprocessing(f,filenumber,outpath)
        filenumber += 1