#coding: utf-8
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

'''全局变量定义
THREAD         线程数量
BUF_SIZE       缓存区大小
DATA_DIR       数据集根目录，对于与处理之后图片应当为，/home/kesci/work/first_round/
TRAIN_LIST     训练集txt索引
VAL_LIST      val集txt索引
class_dim      softmax对应分类数量
num_epochs     迭代pass次数

img_mean       图片均值
img_std        图片方差
'''
random.seed(0)
THREAD = 8
BUF_SIZE = 204800
DATA_DIR='/home/kesci/dataset_fusai/train_val/'
TRAIN_LIST='/home/kesci/work/train_daluan.txt'
VAL_LIST='/home/kesci/work/val_daluan.txt'
TEST_LIST='/home/kesci/pairs_id.txt'
model_path='/home/kesci/work/model_sphereface/'
class_dim=8989
img_mean=np.array([0.546,0.422,0.362]).reshape((3,1,1))
img_std=np.array([0.127,0.112,0.115]).reshape((3,1,1))
pretrained_model='/home/kesci/work/early_stop/persist/8/'

num_epochs=100
total_images=696800
Batch_size=512
Epochs=[5,10,40]
base_lr =0.01
base_lr_decay=0.3
embedding_dim=512


'''sphere20_net网络结构
输入变量：
        img            image，张量形式
        embedding_dim  feature维度 
输出变量：
        pool           图片编码特征，用于提取人脸特征，维度为2048
        out            最后一层全连接层输出，用于计算softmaxloss，维度为class_dim
'''
def sphere20_net(img,embedding_dim):
    def conv_bn_layer(ipt,
                    num_filters,
                    filter_size,
                    stride,
                    param_attr,
                    act_):
        
        conv = fluid.layers.conv2d(input=ipt,
                                    num_filters=num_filters,
                                    filter_size=filter_size,
                                    stride=stride,
                                    padding=1,
                                    param_attr=param_attr)
        
        if act_=='prelu':
            return fluid.layers.prelu(fluid.layers.batch_norm(input=conv,act=None),'channel')
        return fluid.layers.batch_norm(input=conv,act=act_)
        
    def resnet_block(ipt,blocks):
        n = len(blocks)
        w_param1_attrs=fluid.ParamAttr(learning_rate=1,
                                    initializer=blocks[0]['w_init'])
        conv1 = conv_bn_layer(ipt=ipt,
                                num_filters=blocks[0]['filters'],
                                filter_size=blocks[0]['kernel_size'],
                                stride=blocks[0]['strides'],
                                param_attr=w_param1_attrs,
                                act_='relu')
        output = conv1
        for i in range(1,n):
            w_param_attrs=fluid.ParamAttr(learning_rate=1,
                                        initializer=blocks[i]['w_init'])
            output = conv_bn_layer(ipt=output,
                                    num_filters=blocks[i]['filters'],
                                    filter_size=blocks[i]['kernel_size'],
                                    stride=blocks[i]['strides'],
                                    param_attr=w_param_attrs,
                                    act_='relu')
        return fluid.layers.elementwise_add(x=conv1,y=output)
    
    res1_3=[
        {'filters':64, 'kernel_size':3, 'strides':2, 'w_init':fluid.initializer.Xavier(), 'padding':'same'},
        {'filters':64, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
        {'filters':64, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
    ]

    res2_3=[
        {'filters':128, 'kernel_size':3, 'strides':2, 'w_init':fluid.initializer.Xavier(), 'padding':'same'},
        {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
        {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
    ]

    res2_5=[
        {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
        {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
    ]

    res3_3=[
        {'filters':256, 'kernel_size':3, 'strides':2, 'w_init':fluid.initializer.Xavier(), 'padding':'same'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
    ]

    res3_5=[
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
    ]

    res3_7=[
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
    ]

    res3_9=[
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
    ]

    res4_3=[
        {'filters':512, 'kernel_size':3, 'strides':2, 'w_init':fluid.initializer.Xavier(), 'padding':'same'},
        {'filters':512, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
        {'filters':512, 'kernel_size':3, 'strides':1, 'w_init':fluid.initializer.Normal(), 'padding':'same'},
    ]
    output = img
    for suffix,blocks in zip(('1','2','2','3','3','3','3','4'),
                            (res1_3,res2_3,res2_5,res3_3,res3_5,res3_7,res3_9,res4_3)):
        output = resnet_block(output,blocks)
        print output.shape
    f_w_param_attr = fluid.ParamAttr(learning_rate=1,
                                    initializer=fluid.initializer.Xavier())
    feature = fluid.layers.fc(input=output,size=embedding_dim,param_attr=f_w_param_attr)
    
    return feature



    
'''损失函数部分
创建了softmaxloss与centerloss，由于paddle本身原因，centerloss部分无法使用
输入参数：
    fc1         pooling层输出，用作人脸图片编码与centerloss
    fc2         全连接层输出，用作softmaxloss
    label       输入标签
    lmbda       centerloss权重系数，默认0.01
    alpha       配置参数，默认0.1
    num_class   softmax分类数量，默认10575
'''

def total_loss(fc1,label,lmbda=0.01,alpha=0.1):
    '''
    #******计算centerloss**********************************
    len_features=fc1.shape()[1]
    centers=fluid.get_var(name='centers_global')
    
    labels=fluid.layers.reshape(x=label,shape=[-1])
    #获取当前batch每个样本对应的中心
    centers_batch=fluid.layers.gather(centers,labels)
    #计算center loss
    center_cost=fluid.layers.square_error_cost(input=features,label=centers_batch)
    avg_center_cost=fluid.layers.mean(center_cost)
    #更新中心
    diff=(1-alpha)*(centers_batch-features)
    centers=fluid.layers.scatter(x=centers,ids=labels,updates=diff)
    '''
    fc_drop=fluid.layers.dropout(fc1, dropout_prob=0.7)
    fc2=fluid.layers.fc(input=fc_drop,
                        size=class_dim,
                        act='softmax',
                        param_attr=fluid.param_attr.ParamAttr(
                            initializer=fluid.initializer.Xavier()))
                            #Uniform(-0.05,0.05)
    #******计算softmaxloss**********************************
    soft_cost=fluid.layers.cross_entropy(input=fc2,label=label)
    avg_soft_cost=fluid.layers.mean(x=soft_cost)

    #******加权softmaxloss与centerloss
    #avg_cost=avg_soft_cost+avg_center_cost*lmbda
    #return avg_cost, centers
    return fc2,avg_soft_cost

'''reader与图片预处理
1.对图片进行对比度、亮度、饱和度等随机处理
2.图片进行归一化操作，去均值、方差
'''
# 对图片进行对比度、亮度、饱和度等随机处理
def distort_color(img):
    def random_brightness(img,lower=0.5,upper=1.5):
        e=random.uniform(lower,upper)
        return ImageEnhance.Brightness(img).enhance(e)
        
    def random_contrast(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)
        
    ops = [random_brightness, random_contrast, random_color]
    random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img

#图片归一化    
def process_image(sample, mode, color_jitter):
    img_path = sample[0]

    img = Image.open(img_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if random.randint(0, 1) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'val':
        return img, sample[1]
    elif mode == 'test':
        return [img]        

def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False):
    def reader():
        with open(file_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                if mode == 'train' or mode == 'val':
                    img_path, label = line.split()
                    img_path = os.path.join(DATA_DIR, img_path)
                    yield img_path, int(label)
                elif mode == 'test':
                    img_path = os.path.join('/home/kesci/dataset_fusai/test/', line+'.jpg')
                    yield [img_path]

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)
    
def reader_train(file_list=TRAIN_LIST):
    return _reader_creator(
        file_list, 'train', shuffle=True, color_jitter=True)

def reader_val(file_list=VAL_LIST):
    return _reader_creator(file_list, 'val', shuffle=False)

def reader_test(file_list=TEST_LIST):
    return _reader_creator(file_list, 'test', shuffle=False)


'''训练器
'''
def train(model_dir=model_path,pretrained_model=pretrained_model):
    image=fluid.layers.data(name='image',shape=[3,112,96],dtype='float32')
    label=fluid.layers.data(name='label',shape=[1],dtype='int64')
    out1=sphere20_net(image,embedding_dim)
    out2,cost=total_loss(out1,label,lmbda=0.01,alpha=0.1)
    acc_top1=fluid.layers.accuracy(input=out2,label=label,k=1)
    acc_top5=fluid.layers.accuracy(input=out2,label=label,k=5)
    
    test_program = fluid.default_main_program().clone(for_test=True)
    
    step = int(total_images / Batch_size + 1)
    bd = [step * e for e in Epochs]
    lr = []
    lr = [base_lr * (base_lr_decay**i) for i in range(len(bd) + 1)]
    optimizer = fluid.optimizer.Momentum(learning_rate=fluid.layers.piecewise_decay(boundaries=bd, values=lr),
                                        momentum=0.9,
                                        regularization=fluid.regularizer.L2Decay(6e-3))
    opts=optimizer.minimize(cost)   
    
    fluid.memory_optimize(fluid.default_main_program())
    exe = fluid.Executor(fluid.CUDAPlace(0))
    exe.run(program=fluid.default_startup_program())
    
    if pretrained_model:
        print 'load model from: \n',pretrained_model
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)
    
    train_batch_size=Batch_size
    val_batch_size=Batch_size
    
    train_reader=paddle.batch(reader_train(),batch_size=train_batch_size)
    test_reader=paddle.batch(reader_val(),batch_size=val_batch_size)
    
    
    feeder = fluid.DataFeeder(place=fluid.CUDAPlace(0), feed_list=[image, label])
    
    train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=cost.name)
    min_loss=1.2
    fetch_list=[cost.name,acc_top1.name,acc_top5.name]
    for pass_id in range(num_epochs):
        train_info=[[],[],[]]
        test_info=[[],[],[]]
        for batch_id, data in enumerate(train_reader()):
            loss,acc1,acc5=train_exe.run(fetch_list,feed=feeder.feed(data))
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)
            if batch_id %100==0:
                print("pass {0}, trainbatch {1}, loss {2}, acc1 {3}, acc5 {4}".format(pass_id, batch_id, loss, acc1, acc5))
                sys.stdout.flush()
        
        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        cnt=0
        for test_batch_id, data in enumerate(test_reader()):
            loss,acc1,acc5=exe.run(test_program,
                                       fetch_list=fetch_list,
                                       feed=feeder.feed(data))
            loss = np.mean(loss)
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)
            test_info[0].append(loss * len(data))
            test_info[1].append(acc1 * len(data))
            test_info[2].append(acc5 * len(data))
            cnt+=len(data)
            #if test_batch_id % 10 == 0:
            #    print("Pass {0},testbatch {1},loss {2},acc1 {3}, acc5 {4}".format(pass_id, test_batch_id, loss, acc1,acc5))
            #    sys.stdout.flush()
        test_loss = np.sum(test_info[0]) / cnt
        test_acc1 = np.sum(test_info[1]) / cnt 
        test_acc5 = np.sum(test_info[2]) / cnt
        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3},\n\t\t test_loss {4}, test_acc1 {5}, train_acc5 {6}"
                            .format(pass_id,train_loss,train_acc1,train_acc5,test_loss,test_acc1,test_acc5))
        sys.stdout.flush()
        if(min_loss>test_loss):
            min_loss=test_loss
            # 保存预测的模型
            model_dir1 = model_dir + 'infer/'
            model_dir2 = model_dir + 'persist/'
            model_path_save1=os.path.join(model_dir1,str(pass_id))
            model_path_save2=os.path.join(model_dir2,str(pass_id))

            if not os.path.isdir(model_path_save1):
                os.makedirs(model_path_save1)
            if not os.path.isdir(model_path_save2):
                os.makedirs(model_path_save2)
            fluid.io.save_inference_model(model_path_save1,['image'],[out1],exe)
            fluid.io.save_persistables(exe,model_path_save2)
            print ("Save pass {0}, min_loss {1}".format(pass_id,min_loss))
        if(pass_id==5 or pass_id==10 or pass_id==15 or pass_id==20):

            model_dir2 = model_dir + 'persist/'
            model_path_save2=os.path.join(model_dir2,str(pass_id))

            if not os.path.isdir(model_path_save2):
                os.makedirs(model_path_save2)
            fluid.io.save_persistables(exe,model_path_save2)
            print ("Save pass {0}, test_loss {1}".format(pass_id,test_loss))
    '''
    #测试结果
    fetch_list = [out1.name,input_x_sqrt.name,input_x_norm.name]
    for batch_id, data in enumerate(train_reader()):
        opt1,opt2,opt3=train_exe.run(fetch_list,feed=feeder.feed(data))
        print 'out1: ', opt1
        print 'input_x_sqrt: ', opt2
        print 'input_x_norm: ',opt3
    '''
    
        
train(model_dir=model_path,pretrained_model=pretrained_model)  