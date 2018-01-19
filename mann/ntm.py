#!/usr/bin/env python3
# coding=utf-8

import chainer


class NTM(chainer.chain):

    def __init__(self):
        super(NTM, self).__init__()
        with self.init_scope():
            pass

    def __call__(self, x, t):
        return
