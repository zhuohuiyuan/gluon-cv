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


class spherenet(gluon.HybridBlock):
    def __init__(self,class):

