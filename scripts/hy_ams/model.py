# -*- coding:utf-8 -*-
"""
@author:William
@file:model.py
@time:2018/7/13:03 PM
"""
import mxnet as mx
import numpy as np
import math
from mxnet.base import  _Null
from six.moves import xrange

def conv_main(data,units,filters,workspace):
    body=data
    for i in xrange(len(units)):
        f=filters[i]
        _weight=mx.symbol.Variable('conv%d_%d_weight' % (i+1,1),lr_mult=1.0)
        _bias=mx.symbol.Variable('conv%d_%d_bias' % (i+1,1),lr_mult=2.0,wd_mult=0.0)
        body=mx.symbol.Convolution(data=body,weight=_weight, bias=_bias,num_filter=f,kernel=(3,3),stride=(2,2),pad=(1,1),name='conv%d_%d' % (i+1,1),workspace=workspace)
        body=mx.symbol.LeakyReLU(data=body,act_type='prelu',name='relu%d_%d' % (i+1,1))
        idx=2
        for j in xrange(units[i]):
            _body=mx.symbol.Convolution(data=body,no_bias=True,num_filter=f,kernel=(3,3),stride=(1,1),pad=(1,1),name='conv%d_%d' % (i+1,idx),workspace=workspace)
            _body=mx.symbol.LeakyReLU(data=_body,act_type='prelu',name='relu%d_%d' % (i+1,idx))
            idx+=1
            _body=mx.symbol.Convolution(data=_body,no_bias=True,num_filter=f,kernel=(3,3),stride=(1,1),pad=(1,1),name='conv%d_%d' % (i+1,idx),workspace=workspace)
            _body=mx.symbol.LeakyReLU(data=_body,act_type='prelu',name='relu%d_%d' % (i+1,idx))
            idx+=1
            body=body+_body

    return body

def get_symbol(num_classes,num_layers,conv_workspace=256,**kwargs):
    if num_layers==64:
        units=[3,8,16,3]
        filters=[64,128,256,512]
    elif num_layers==20:
        units=[1,2,4,1]
        filters = [64, 128, 256, 512]
        #filters=[64,256,512,1024]
    elif num_layers==36:
        units=[2,4,8,2]
        filters = [64, 128, 256, 512]
        # filters=[64,256,512,1024]
    elif num_layers==60:
        units=[3,8,14,3]
        filters = [64, 128, 256, 512]
    elif num_layers==104:
        units=[3,8,36,3]
        filters = [64, 128, 256, 512]
        # filters=[64,256,512,1024]
    data=mx.symbol.Variable('data')
    data=data-127.5
    data=data*0.0078125
    body=conv_main(data=data,units=units,filters=filters,workspace=conv_workspace)

    _weight=mx.symbol.Variable('fc1_weight',lr_mult=1.0)
    _bias=mx.symbol.Variable('fc1_bias',lr_mult=2.0,wd_mult=0.0)
    fc1=mx.symbol.FullyConnected(data=body,weight=_weight,bias=_bias,num_hidden=num_classes,name='fc1')
    return fc1

def init_weights(sym,data_shape_dict,num_layers):
    arg_name=sym.list_arguments()
    aux_name=sym.list_auxiliary_states()
    






























