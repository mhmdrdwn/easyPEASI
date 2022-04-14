#!/usr/bin/env python
# -*- coding: utf-8 -*-

__modifiedby__ = 'Mohamed Radwan'
__originalauthor__ = 'David Nahmias'
__credits__ = ['David Nahmias']
__maintainer__ = 'Mohamed Radwan'

import logging
import time
from copy import copy
import sys

from collections import Counter
import random
import numpy as np
from numpy.random import RandomState
import resampy
from torch import optim
import torch.nn.functional as F
import torch as th
from torch.nn.functional import elu
from torch import nn

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              MisclassMonitor)
from braindecode.experiments.stopcriteria import MaxEpochs
#from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import var_to_np
from braindecode.torch_ext.functions import identity


def splitDataRandom(allData,allLabels,setNum=0,shuffle=0):
    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum

    labels0 = allLabels[allLabels == 0]
    labels1 = allLabels[allLabels == 1]
    data0 = np.array(allData)[allLabels == 0]
    data1 = np.array(allData)[allLabels == 1]

    fullRange = list(range(numberEqSamples))
    random.shuffle(fullRange)

    testIndices = fullRange[trainSamplesNum:]
    trainIndices = fullRange[:trainSamplesNum]
    
    allDataTrain = np.concatenate((data0[trainIndices],data1[trainIndices]),axis=0)
    allLabelsTrain = np.concatenate((labels0[trainIndices],labels1[trainIndices]),axis=0)

    allDataTest = np.concatenate((data0[testIndices],data1[testIndices]),axis=0)
    allLabelsTest = np.concatenate((labels0[testIndices],labels1[testIndices]),axis=0)

    return allDataTrain, allLabelsTrain, allDataTest, allLabelsTest


def create_set(X, y, inds):
    """
    X list and y nparray
    :return: 
    """
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return SignalAndTarget(new_X, new_y)


class TrainValidTestSplitter(object):
    def __init__(self, n_folds, i_test_fold, shuffle):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.rng = RandomState(404)
        self.shuffle = shuffle

    def split(self, X, y,):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        test_inds = folds[self.i_test_fold]
        valid_inds = folds[self.i_test_fold - 1]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        test_set = create_set(X, y, test_inds)

        return train_set, valid_set, test_set


class TrainValidSplitter(object):
    def __init__(self, n_folds, i_valid_fold, shuffle):
        self.n_folds = n_folds
        self.i_valid_fold = i_valid_fold
        self.rng = RandomState(404)
        self.shuffle = shuffle

    def split(self, X, y):
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        valid_inds = folds[self.i_valid_fold]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, valid_inds)
        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        return train_set, valid_set

    
def preprocess(config):
    preproc_functions = []
    preproc_functions.append(
        lambda data, fs: (data[:, int(config.sec_to_cut * fs):-int(
            config.sec_to_cut * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (data[:, :int(config.duration_recording_mins * 60 * fs)], fs))
    if config.max_abs_val is not None:
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -config.max_abs_val, config.max_abs_val), fs))

    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                config.sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               config.sampling_freq))

    if config.divisor is not None:
        preproc_functions.append(lambda data, fs: (data / config.divisor, fs))
        
    return preproc_functions

    
def data_loader(config):
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    
    preproc_functions = preprocess(config)
    
    dataset = DiagnosisSet(n_recordings=config.n_recordings,
                           max_recording_mins=config.max_recording_mins,
                           preproc_functions=preproc_functions,
                           data_folders=config.data_folders,
                           train_or_eval='train',
                           sensor_types=config.sensor_types)
    
    if config.test_on_eval:
        test_recording_mins = config.duration_recording_mins
        test_preproc_functions = copy(preproc_functions)
        test_preproc_functions[1] = lambda data, fs: (
            data[:, :int(test_recording_mins * 60 * fs)], fs)
        test_dataset = DiagnosisSet(n_recordings=config.n_recordings,
                                max_recording_mins=None,
                                preproc_functions=test_preproc_functions,
                                data_folders=config.data_folders,
                                train_or_eval='val',
                                sensor_types=config.sensor_types)

    data, labels = dataset.load()
    X,y,test_X,test_y = splitDataRandom(data, labels,shuffle=0)
    
    if config.test_on_eval:
        max_shape = np.max([list(x.shape) for x in test_X], axis=0)
    if not config.test_on_eval:
        splitter = TrainValidTestSplitter(config.n_folds, config.i_test_fold,
                                          shuffle=config.shuffle)
        train_set, valid_set, test_set = splitter.split(X, y)
        
    else:
        splitter = TrainValidSplitter(config.n_folds, i_valid_fold=config.i_test_fold,
                                          shuffle=config.shuffle)
        train_set, valid_set = splitter.split(X, y)
        test_set = SignalAndTarget(test_X, test_y)
        
    return train_set, valid_set, test_set

