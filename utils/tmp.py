#!/usr/bin/env python3
# coding=utf-8

import argparse
import glob
import os
import sys
import random

import chainer
import numpy as np


class SamplingDataset(chainer.dataset.DatasetMixin):

    def __init__(self,
                 dataset,
                 num_class,
                 num_data_per_class,
                 num_sample_class,
                 num_sample_per_class,
                 rand_state=True):
        self.dataset = chainer.datasets.ImageDataset(dataset)
        self.num_class = num_class
        self.num_data_per_class = num_data_per_class
        self.num_sample_class = num_sample_class
        self.num_sample_per_class = num_sample_per_class
        self.rand_state = rand_state

        self.sampleddataset = None
        self.resampling()

    def __call__(self):
        return self.sampleddataset

    def __len__(self):
        return len(self.sampleddataset)

    def resampling(self):
        class_indices = random.sample(range(self.num_class),
                                      self.num_sample_class) * self.num_sample_per_class
        per_class_indices = np.asarray([
            random.sample(range(self.num_data_per_class), self.num_sample_per_class)
            for _ in range(self.num_sample_class)
        ]).T
        labels = list(range(self.num_sample_class)) * self.num_data_per_class
        data_indices = np.asarray(class_indices) * self.num_data_per_class + per_class_indices.flat
        if self.rand_state:
            order = np.random.permutation(self.num_sample_class * self.num_sample_per_class)
        else:
            order = np.arange(self.num_sample_class * self.num_sample_per_class)
        print(order)
        self.sampleddataset = list(
            zip(self.dataset[data_indices[order]], np.asarray(labels)[order]))

    def get_example(self, i):
        return self.sampleddataset[i]


def make_sampling_dataset_for_omniglot_traindata(basedir,
                                                 num_sample_class=5,
                                                 num_sample_per_class=10):
    paths = glob.glob(os.path.join(basedir, "*", "*", "*"))
    num_class = len(glob.glob(os.path.join(basedir, "*", "*")))

    return SamplingDataset(paths, num_class, 20, num_sample_class, num_sample_per_class)


class RandomSampleIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, num_class, num_samples_per_class):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_class = num_class
        self.num_samples = num_samples_per_class

        self.epoch = 0
        self.iteration = 0
        self._previous_epoch_detail = -1.
        self.is_new_epoch = False

    def __next__(self):
        self.epoch += 1
        self.iteration += 1
        self.is_new_epoch = True
        order = np.random.permutation(len(self.dataset))
        batch = [self.dataset[index] for index in order]

        return list(zip(batch, range(self.num_class)))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return float(self.epoch)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        try:
            self._previous_epoch_detail = serializer('previous_epoch_detail',
                                                     self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.


if __name__ == "__main__":
    sys.path.append("../")
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', '-d', default="../data/omniglot/images_background", help='')
    args = parser.parse_args()

    sd = make_sampling_dataset_for_omniglot_traindata(args.basedir)
    print(sd())
