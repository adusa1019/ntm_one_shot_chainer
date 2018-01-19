#!/usr/bin/env python3
# coding=utf-8

import argparse
import glob
import os
import random
import sys

import chainer
import numpy as np


def transform(in_data):
    """
    画像の白黒を反転する
    in_data: 画素値が0.0~1.0に正規化されたグレースケール画像
    """
    return 1.0 - in_data


class SamplingDataset(chainer.dataset.DatasetMixin):

    def __init__(
            self,
            dataset,
            num_class,
            num_data_per_class,
            num_sample_class,
            num_sample_per_class,
    ):
        self.dataset = chainer.datasets.TransformDataset(
            chainer.datasets.ImageDataset(dataset), transform)
        self.num_class = num_class
        self.num_data_per_class = num_data_per_class
        self.num_sample_class = num_sample_class
        self.num_sample_per_class = num_sample_per_class

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
        labels = list(range(self.num_sample_class)) * self.num_sample_per_class
        data_indices = np.asarray(class_indices) * self.num_data_per_class + per_class_indices.flat
        self.sampleddataset = list(zip(self.dataset[data_indices], labels))

    def get_example(self, i):
        return self.sampleddataset[i]


def make_sampling_dataset_for_omniglot_traindata(basedir,
                                                 num_sample_class=5,
                                                 num_sample_per_class=10):
    paths = glob.glob(os.path.join(basedir, "*", "*", "*"))
    num_class = len(glob.glob(os.path.join(basedir, "*", "*")))

    return SamplingDataset(paths, num_class, 20, num_sample_class, num_sample_per_class)


class RandomSampleIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, iter_per_epoch=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.iter_per_epoch = iter_per_epoch
        self.shuffle = shuffle

        self.dataset.resampling()
        self.reset()

    def __next__(self):
        self.prev_epoch_detail = self.epoch_detail
        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        if self.order is None:
            batch = self.dataset[i:i_end]
        else:
            batch = self.dataset[self.order[i:i_end]]

        if i_end >= N:
            self.iter += 1
            if self.order is not None:
                np.random.shuffle(self.order)
            if self.iter < self.iter_per_epoch:
                rest = i_end - N
                if rest > 0:
                    if self.order is None:
                        batch.extend(self.dataset[:rest])
                    else:
                        batch.extend(self.dataset[self.order[:rest]])
                self.current_position = rest
            else:
                self.epoch += 1
                self.iter = 0
                self.is_new_epoch = True
                self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self.prev_epoch_detail < 0:
            return None
        return self.prev_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position', self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self.order is not None:
            serializer('order', self.order)
        try:
            self.prev_epoch_detail = serializer('previous_epoch_detail', self.prev_epoch_detail)
        except KeyError:
            self.prev_epoch_detail = self.epoch + (
                self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self.prev_epoch_detail = max(self.prev_epoch_detail, 0.)
            else:
                self.prev_epoch_detail = -1.

    def reset(self):
        self.epoch = 0
        self.iter = 0
        self.prev_epoch_detail = -1
        self.is_new_epoch = False
        self.current_position = 0
        if self.shuffle:
            self.order = np.random.permutation(len(self.dataset))
        else:
            self.order = None


if __name__ == "__main__":
    sys.path.append("../")
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', '-d', default="../data/omniglot/images_background", help='')
    args = parser.parse_args()

    sd = make_sampling_dataset_for_omniglot_traindata(args.basedir)
    print(np.shape(sd()))
    rsi = RandomSampleIterator(sd, 16, 16)
    while rsi.epoch < 100:
        batch = rsi.next()
        print(rsi.epoch)
        print(np.asarray(batch)[:, 1])
