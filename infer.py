# -*- coding: utf-8 -*-
import paddle
import paddle.fluid as fluid
import math
import paddle.fluid.layers.ops as ops
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import os,sys
import random
import functools
import numpy as np
from PIL import Image, ImageEnhance
import math
import csv

THREAD = 8
TEST_LIST='/home/kesci/pairs_id.txt'
model_path='/home/kesci/work/early_stop/infer/10/'
class_dim=8989
img_mean=np.array([0.546,0.422,0.362]).reshape((3,1,1))
img_std=np.array([0.127,0.112,0.115]).reshape((3,1,1))


def infer(read_path,mirror,save_path):
    place=fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        f=open(read_path,'r')
        fw=open(save_path,'a')
        for line in f.readlines():
            line=line.strip()
            img_path='/home/kesci/dataset_fusai/test/'+line+'.jpg'
            img1=Image.open(img_path)
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if mirror==True:
                img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img1 = np.array(img1).astype('float32').transpose((2, 0, 1)) / 255
            img1 -= img_mean
            img1 /= img_std
            img1 = img1[np.newaxis, :]
            results = exe.run(inference_program,
                              feed={feed_target_names[0]: img1},
                              fetch_list=fetch_targets)
            
            fe=np.array(results).flatten()
            fw.write(' '.join(str(a) for a in fe))
            fw.write('\n')
        fw.close()
        f.close()
        
infer(TEST_LIST,True,save_path='/home/kesci/features_mirror.txt')
infer(TEST_LIST,False,save_path='/home/kesci/features.txt')  