# -*- coding:utf-8 -*-
"""
@author:William
@file:gluon_model.py
@time:2018/7/311:31 PM
"""
from mxnet import nd,gluon
import mxnet as mx
from mxnet.gluon.loss import _reshape_like
import numpy as np

class Block1(gluon.nn.HybridBlock):

    def __init__(self,channels,**kwargs):
        super(Block1,self).__init__(**kwargs)
        with self.name_scope():
            self.conv1=gluon.nn.Conv2D(channels,kernel_size=3,stride=2,padding=1)
            self.prelu1=gluon.nn.LeakyReLU(alpha=0.25)
            self.conv2=gluon.nn.Conv2D(channels,kernel_size=3,stride=1,padding=1)
            self.prelu2=gluon.nn.LeakyReLU(alpha=0.25)
            self.conv3=gluon.nn.Conv2D(channels,kernel_size=3,strides=1,padding=1)
            self.prelu3=gluon.nn.LeakyReLU(alpha=0.25)

    def hybrid_forward(self, F, x):
        x=self.prelu1(self.conv1(x))
        return x+self.prelu3(self.conv3(self.prelu2(self.conv2(x))))

class Block2(gluon.nn.HybridBlock):

    def __init__(self,channels,**kwargs):
        super(Block2, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1=gluon.nn.Conv2D(channels,kernel_size=3,strides=1,padding=1)
            self.prelu1=gluon.nn.LeakyReLU(alpha=0.25)
            self.conv2=gluon.nn.Conv2D(channels,kernel_size=3,strides=1,padding=1)
            self.prelu2=gluon.nn.LeakyReLU(alpha=0.25)

    def hybrid_forward(self, F, x):
        x=self.prelu1(self.conv1(x))
        return x+self.prelu2(self.conv2(x))

class spherenet20(gluon.HybridBlock):
    def __init__(self,classes=10574,feature=False,m=4,**kwargs):
        super(spherenet20, self).__init__()
        self.classes=classes
        self.feature=feature
        self.m=m
        self.net_sequence=gluon.nn.HybridSequential()
        self.get_feature=False
        #input=B*3*112*96
        with self.net_sequence.name_scope():
            self.net_sequence.add(Block1(channels=64),
                                  Block1(channels=128),
                                  Block2(channels=128),
                                  Block1(channels=256),
                                  Block2(channels=256),
                                  Block2(channels=256),
                                  Block2(channels=256),
                                  Block1(channels=512)
            )
            self.fc5=gluon.nn.Dense(512)
    def hybrid_forward(self, F, x):
        x=self.net_sequence(x)
        return self.fc5(x)
