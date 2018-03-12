# -*- coding: utf-8 -*-
import os 
import matplotlib as mpl 
import matplotlib.image as mping
from matplotlib import pyplot as plt 

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn , utils
from mxnet import autograd
import numpy as np

epochs = 2
batch_size = 4
latent_z_size = 100

def try_gpu():
    ctx = mx.gpu()
    try:
        _ = nd.array([1],ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
ctx = try_gpu()

lr = .0002
beta1 = .5

data_path = ('face')
img_list = []
#read data ,transform to ndarray , get the training data 
for _,_,files in os.walk(data_path):
    for file_name in files:
        if not file_name.endswith('.jpg'):
            continue
        img_dir = os.path.join(data_path , file_name)
        img_arr = mx.image.imread(img_dir)
        #from HWC transpose to format of CHW 
        img_arr = nd.transpose(img_arr,(2,0,1))
        #normalize pixel to [-1,1] to fit the scale of input 
        img_arr = (img_arr.astype(np.float32)/127.5-1)
        #add one dimension (from 3 to 4)
        img_arr = nd.array(img_arr.reshape((1,)+img_arr.shape))
        img_list.append(img_arr)
train_data = mx.io.NDArrayIter(data = nd.concatenate(img_list),batch_size=batch_size)

#function to visualize image
def visualize(img_arr):
    #from CHW transpose to format HWC
    plt.imshow(((img_arr.asnumpy().transpose(1,2,0)+1.0)*127.5).astype(np.uint8)) 
    plt.axis('off')
 
#show 9 images
fig = plt.figure(figsize=(9,9))
for i in range(9):
    plt.subplot(3,3,i+1)
    visualize(img_list[i][0])
plt.show()

#define the networks
#=============discriminator============
netD = nn.Sequential()
with netD.name_scope():
    netD.add(nn.Conv2D(channels=128,kernel_size=6,strides=2,padding=2))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    netD.add(nn.Conv2D(channels=256,kernel_size=6,strides=2,padding=2))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    netD.add(nn.Conv2D(channels=512,kernel_size=6,strides=2,padding=2))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    netD.add(nn.Conv2D(channels=1024,kernel_size=6,strides=2,padding=2))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(.2))
    netD.add(nn.Conv2D(channels=1,kernel_size=6))
    
#===============generator==================
netG = nn.Sequential()
with netG.name_scope():
    netG.add(nn.Conv2DTranspose(channels=1024,kernel_size=6))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    netG.add(nn.Conv2DTranspose(channels=512,kernel_size=6,strides=2,padding=2))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    netG.add(nn.Conv2DTranspose(channels=256,kernel_size=6,strides=2,padding=2))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    netG.add(nn.Conv2DTranspose(channels=128,kernel_size=6,strides=2,padding=2))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='relu'))
    netG.add(nn.Conv2DTranspose(channels=3,kernel_size=6,strides=2,padding=2))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation(activation='tanh'))
    
#loss , initialization , trainer 
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

netG.initialize(mx.init.Normal(.02),ctx=ctx)
netD.initialize(mx.init.Normal(.02),ctx=ctx)

trainerG = gluon.Trainer(netG.collect_params(),'adam',{'learning_rate':lr,'beta1':beta1})
trainerD = gluon.Trainer(netD.collect_params(),'adam',{'learning_rate':lr,'beta1':beta1})

#training loop
import time
import logging 

real_label = nd.ones((batch_size,),ctx=ctx)
fake_label = nd.zeros((batch_size,),ctx=ctx)

def eveluate(pred , label):
    pred = pred.flatten()
    label = label.flatten()
    return ((pred>.5) == label).mean()
metric = mx.metric.CustomMetric(eveluate)#custom metric

logging.basicConfig(level=logging.DEBUG)
his_acc = []
his_errD = []
his_errG = []
#funtion to save the netG
path = 'his_params'
if not os.path.exists(path):
    os.makedirs(path)
def save_params(net,epoch,path):
    file_path = os.path.join(path,str(epoch))
    net.save_params(file_path)
    
for epoch in range(epochs):
    start_time = time.time()
    train_data.reset()
    for batch in train_data:
        #========G fixed , train D,maxmize log(D(x)) + los(1-D(G(z)))======
        data = batch.data[0].as_in_context(ctx)
        latent_z = mx.nd.random_normal(0,1,shape=(batch_size,latent_z_size,1,1),ctx=ctx)
        
        with autograd.record():
            output = netD(data).reshape((-1,1))
            errD_real = loss(output,real_label)
            metric.update([output,],[real_label,])
            
            fake = netG(latent_z)
            output_fake = netD(fake).reshape((-1,1))
            errD_fake = loss(output_fake,fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([output_fake],[fake_label])
        
        trainerD.step(batch_size)
            
        #=======D fixed , train G, maxmize log(D(G(z)))============
        with autograd.record():
            fake = netG(latent_z)
            output_fake = netD(fake).reshape((-1,1))
            errG = loss(output_fake,real_label)
            errG.backward()
        
        trainerG.step(batch_size)
      
    end_time = time.time() 
    _,acc = metric.get()
    his_acc.append(acc)
    his_errD.append(nd.mean(errD).asscalar())
    his_errG.append(nd.mean(errG).asscalar())
    save_params(netG,epoch,path)
    logging.info('epoch: %i ; discriminator loss:%f ; generator loss:%f ; training acc:%f ; time:%f'
                 %(epoch , nd.mean(errD).asscalar(),nd.mean(errG).asscalar(),acc,end_time-start_time))
    metric.reset()
    if (0 < epoch < 10) or ((epoch % 10) == 0):
        fig = plt.figure(figsize=(12,12))
        for i in range(4):
            latent_z = mx.nd.random_normal(0,1,shape=(4,latent_z_size,1,1),ctx=ctx)
            fake = netG(latent_z)
            plt.subplot(2,2,i+1)
            visualize(fake[i])
        plt.show()    

#plot the data
x_axis = np.linspace(0,epochs,len(his_acc))
plt.figure(figsize=(12,12))
plt.plot(x_axis,his_acc,label='accuracy')
plt.plot(x_axis,his_errD,label='error of Discriminator')
plt.plot(x_axis,his_errG,label='error of Generator')
plt.xlabel('epoch')
plt.legend()
plt.show()


        
    
    
