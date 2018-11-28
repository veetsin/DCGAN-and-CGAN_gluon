#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:04:47 2018
cgan_mnist_20181101005
goal:get images that can be generated with labels

@author: veetsin
"""

# -*- coding: utf-8 -*-
import os 
from matplotlib import pyplot as plt 

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn 
from mxnet import autograd
import numpy as np

serial_number = 0
name = 'cgan_mnist_20181101005'

epochs = 121
batch_size = 256
latent_z_size = 100

path_data = name
if not os.path.exists(path_data):
    os.makedirs(path_data)

def try_gpu():
    ctx = mx.gpu()
    try:
        _ = nd.array([1],ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
ctx = try_gpu()

lr = .03
beta1 = 0
beta2 = 0.9


##function to visualize image
#def visualize(img_arr):
#    #from CHW transpose to format HWC
#    plt.imshow((img_arr[0][0].transpose(1,2,0)*255).astype(np.uint8)) 
#    plt.axis('off')


def transform(data, label):
#    data = mx.image.imresize(data,33,33)
    return nd.transpose(data.astype(np.float32), (2,0,1))/255 , label.astype(np.float32)
mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)

def show_images(images):
        plt.imshow(images.reshape((28, 28)).asnumpy())
        plt.axis('off')



#====================show original images===============
fig = plt.figure(figsize=(10,10))
for i in range(9):
    data,_ = mnist_train[i]
    plt.subplot(3,3,i+1)
    show_images(data)
plt.show() 

#=============discriminator============
class Discriminator(nn.Block):
    def __init__(self, n_dims=128, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(32, kernel_size=4, strides=2, padding=1)
            self.conv2 = nn.Conv2D(256, kernel_size=4, strides=2, padding=1)
            self.conv3 = nn.Conv2D(512, kernel_size=4)
            self.conv4 = nn.Conv2D(1, kernel_size=4)
            
            self.bn2 = nn.BatchNorm()
            self.bn3 = nn.BatchNorm()
        
    def forward(self, x, y):
        x = nd.concat(x, y, dim=1)
        x = nd.LeakyReLU(self.conv1(x))
        x = nd.LeakyReLU(self.bn2(self.conv2(x)))
        x = nd.LeakyReLU(self.bn3(self.conv3(x)))
        
#        y = nd.expand_dims(y, axis=2)
#        y = nd.expand_dims(y, axis=2)
#        y = nd.tile(y, [4,4])
#        
        x = self.conv4(x)
        
        return x
netD = Discriminator()

#===============generator==================
class Generator(nn.Block):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)
        with self.name_scope():
            self.dense_z = nn.Dense(256)
            self.dense_label = nn.Dense(256)
            self.deconv2 = nn.Conv2DTranspose(256, kernel_size=7)
            self.deconv3 = nn.Conv2DTranspose(32, kernel_size=4, strides=2,padding=1)
            self.deconv4 = nn.Conv2DTranspose(1, kernel_size=4, strides=2,padding=1)
            
            self.bn_z = nn.BatchNorm()
            self.bn_label = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()
            self.bn3 = nn.BatchNorm()
            self.bn4 = nn.BatchNorm()

    
    def forward(self, x, y):
        x = nd.relu(self.bn_z(self.dense_z(x)))
        
        y = nd.expand_dims(y, axis=2)
        y = nd.expand_dims(y, axis=2)
        y = nd.relu(self.bn_label(self.dense_label(y)))
        
        z = nd.concat(x, y, dim=1)
        
        z = z.reshape([z.shape[0],z.shape[1],1,1])
        x = nd.relu(self.bn2(self.deconv2(z)))
        x = nd.relu(self.bn3(self.deconv3(x)))
        x = nd.relu(self.bn4(self.deconv4(x)))

#        x = nd.sigmoid(self.out(z))
        
        return x

netG = Generator()
#class Generator(nn.Block):
#    def __init__(self, n_dims=128, **kwargs):
#        super(Generator, self).__init__(**kwargs)
#        with self.name_scope():
#            self.deconv_z = nn.Conv2DTranspose(n_dims * 2, kernel_size=4)
#            self.deconv_label = nn.Conv2DTranspose(n_dims * 2, kernel_size=4)
#            self.deconv2 = nn.Conv2DTranspose(n_dims * 2, kernel_size=4)
#            self.deconv3 = nn.Conv2DTranspose(n_dims, kernel_size=4, strides=2,padding=1)
#            self.deconv4 = nn.Conv2DTranspose(1,kernel_size=4, strides=2,padding=1)
#            
#            self.bn_z = nn.BatchNorm()
#            self.bn_label = nn.BatchNorm()
#            self.bn2 = nn.BatchNorm()
#            self.bn3 = nn.BatchNorm()
#    
#    
#    def forward(self, x, y):
#        x = nd.relu(self.bn_z(self.deconv_z(x)))
#        
#        y = nd.expand_dims(y, axis=2)
#        y = nd.expand_dims(y, axis=2)
#        y = nd.relu(self.bn_label(self.deconv_label(y)))
#        
#        z = nd.concat(x, y, dim=1)
#        
#        x = nd.relu(self.bn2(self.deconv2(z)))
#        x = nd.relu(self.bn3(self.deconv3(x)))
#        x = nd.tanh(self.deconv4(x))
#        
#        return x
#netG = Generator()

#loss , initialization , trainer 
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

netG.initialize(mx.init.Normal(.02),ctx=ctx)
netD.initialize(mx.init.Normal(.02),ctx=ctx)

trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': beta2})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': beta2})

#training loop
import time
import logging 

real_label = nd.ones([batch_size,],ctx=ctx)
fake_label = nd.zeros([batch_size,],ctx=ctx)


#custom metric
#def eveluate(pred , label):
#    pred = pred.flatten()
#    label = label.flatten()
#    return ((pred>.5) == label).mean()
#metric_real = mx.metric.CustomMetric(eveluate)
#metric_fake = mx.metric.CustomMetric(eveluate)
#logging.basicConfig(level=logging.DEBUG)
#his_acc_real = []
#his_acc_fake = []
#his_errD_real = []
#his_errD_fake = []
#his_errG = []


#funtion to save the net
#path_D = 'his_params_D_CGAN'
path_G = os.path.join(name, 'his_params_G')
#if not os.path.exists(path_D):
#    os.makedirs(path_D)
if not os.path.exists(path_G):
    os.makedirs(path_G)

def save_params(net,epoch,path):
    file_path = os.path.join(path,str(epoch))
    net.save_parameters(file_path)
    

#for batch in train_data:
#    batch = batch
#    break
#data = nd.array(batch[0].asnumpy())
#labels = nd.array(batch[1],ctx=ctx)
#data = data.as_in_context(ctx)
#y = labels.reshape([batch_size,1,1,1])
#y = y.broadcast_to([batch_size,1,28,28])           
#latent_z = mx.nd.random_normal(0,1,shape=(batch_size,latent_z_size,1,1),ctx=ctx)
#y_z = mx.nd.array(np.random.randint(0, 9, size=batch_size))
#y_z = nd.one_hot(y_z, depth=10).as_in_context(ctx)
#
#fake = netG(latent_z,y_z)


#test = netD(data,y)

#function to plot gif of generated images
#path_gif = 'gif_CGAN'
#if not os.path.exists(path_gif):
#    os.makedirs(path_gif)
#import cv2
#def save_gif(img_arr, path , epoch_number):
#    epoch_number = str(epoch_number)+'.jpg'
#    file_path = os.path.join(path,epoch_number) 
#    #from CHW transpose to format HWC
#    cv2.imwrite(file_path , ((img_arr.asnumpy().transpose(1,2,0)+1.0)*127.5).astype(np.uint8))    

for epoch in range(epochs):
    if epoch == int(epoch/4):
        trainerD.set_learning_rate(lr*.5)
        trainerG.set_learning_rate(lr*.5)
    if epoch == int(epoch*3/4):
        trainerD.set_learning_rate(lr*.2)
        trainerG.set_learning_rate(lr*.2)
    start_time = time.time()
#    sum_errD_real = []
#    sum_errD_fake = []
#    sum_errG = []
    for batch in train_data:
        if batch[1].shape[0] != batch_size:
            break
        #========G fixed , train D,maxmize log(D(x)) + log(1-D(G(z)))======
        data = nd.array(batch[0].asnumpy())
        labels = nd.array(batch[1],ctx=ctx)
        data = data.as_in_context(ctx)
        y = labels.as_in_context(ctx).reshape([batch_size,1,1,1])
#        y = labels.reshape([batch_size,1,1,1])
        y = y.broadcast_to([batch_size,1,28,28])       
        latent_z = mx.nd.random_normal(0,1,shape=(batch_size,latent_z_size,1,1),ctx=ctx)
        
        y_z_1 = mx.nd.array(np.random.randint(0, 9, size=batch_size),ctx=ctx)
        y_tem = y_z_1.reshape([256,1])
#        y_z_onehot = nd.one_hot(y_z_1, depth=10).as_in_context(ctx)
        y_z_onehot = y_tem.broadcast_to([256,100])
        y_z = y_z_1.reshape([batch_size,1,1,1])
        y_z = y_z.broadcast_to([batch_size,1,28,28])
        
        with autograd.record():
            
            #print(data.shape)
            output = netD(data,y).reshape((-1,1))
            errD_real = loss(output,real_label)
#           correct metric_real.update([output,],[real_label,])
#            metric_real.update(output,real_label)
            
            fake = netG(latent_z,y_z_onehot)
            output_fake = netD(fake.detach(),y_z).reshape((-1,1))
            errD_fake = loss(output_fake,fake_label)
            errD = errD_real + errD_fake
#            sum_errD_real.append(nd.mean(errD_real).asscalar())
#            sum_errD_fake.append(nd.mean(errD_fake).asscalar())
            errD.backward()
            # correct metric_fake.update([output,],[real_label,])
#            metric_fake.update(output_fake,fake_label)
        
        trainerD.step(batch_size)
        #=======D fixed , train G, maxmize log(D(G(z)))============
        with autograd.record():
            fake = netG(latent_z,y_z_onehot)
            output_fake = netD(fake,y_z).reshape((-1,1))
            errG = loss(output_fake,real_label)
#            sum_errG.append(nd.mean(errG).asscalar())
            errG.backward()
            
        trainerG.step(batch_size)
    
    end_time = time.time() 
#    _,acc_real = metric_real.get()
#    _,acc_fake = metric_fake.get()
#    his_acc_real.append(acc_real)
#    his_acc_fake.append(acc_fake)
#    his_errD_real.append(sum(sum_errD_real)/len(sum_errD_real))
#    his_errD_fake.append(sum(sum_errD_fake)/len(sum_errD_fake))
#    his_errG.append(sum(sum_errG)/len(sum_errG))
#    save_params(netD,epoch,path_D)
    save_params(netG,epoch,path_G)
    logging.info('epoch: %i ;time:%f '%(epoch ,end_time-start_time))
    print('%i epoch ends',(epoch))
#    logging.info('epoch: %i ; discriminator loss of real :%f ;discriminator loss of fake :%f ; generator loss:%f ; time:%f ;acc_real:%f ; acc_fake:%f'
#                 %(epoch , sum(sum_errD_real)/len(sum_errD_real) , sum(sum_errD_fake)/len(sum_errD_fake) , sum(sum_errG)/len(sum_errG) ,end_time-start_time,acc_real,acc_fake))
#    metric_real.reset()
#    metric_fake.reset()
#    if (0 < epoch < 10) or ((epoch % 10) == 0):
#        fig = plt.figure(figsize=(10,10))
#        for i in range(9):
#            latent_z = mx.nd.random_normal(0,1,shape=(1,latent_z_size,1,1),ctx=ctx)
#            fake = netG(latent_z,nd.one_hot(nd.array([9,]),depth=10).as_in_context(ctx))
#            plt.subplot(3,3,i+1)
#            show_images(fake[0])
#        plt.savefig(os.path.join(path_data,str(epoch)+'.png'))
#        plt.show() 
#        
#        
    if (0 < epoch < 10) or ((epoch % 10) == 0):
        fig = plt.figure(figsize=(10,10))
        for i in range(100):
            latent_z = mx.nd.random_normal(0,1,shape=(1,latent_z_size,1,1),ctx=ctx)
            fake = netG(latent_z,(nd.ones([1,100,1,1])*(i//10)).as_in_context(ctx))
            plt.subplot(10,10,i+1)
            show_images(fake[0])
        plt.savefig(os.path.join(path_data,str(epoch)+'.png'))
        plt.show() 
        
#    mx.random.seed(12)
#    latent_z_gif = mx.nd.random_normal(0,1,shape=(batch_size,latent_z_size,1,1),ctx=ctx)
#    fake = netG(latent_z_gif)
#    save_gif(fake[3],path_gif,epoch)
    

  


#plot the data
#x_axis = np.linspace(0,epochs,len(his_acc_real))
#plt.figure(figsize=(15,10))
#plt.plot(x_axis,his_errG,label='error of Generator')
#plt.xlabel('epoch')
#plt.legend()
#plt.show()

#x_axis = np.linspace(0,epochs,len(his_acc_real))
#plt.figure(figsize=(15,10))
#plt.plot(x_axis,his_errD_real,label='error of Discriminating real data')
#plt.plot(x_axis,his_errD_fake,label='error of Discriminating fake data')
#plt.xlabel('epoch')
#plt.legend()
#plt.show()


#plot acc_real and acc_fake seperately
#x_axis = np.linspace(0,epochs,len(his_acc_real))
#plt.figure(figsize=(15,10))
#plt.plot(x_axis,his_acc_real,label='acc_real')
#plt.plot(x_axis,his_acc_fake,label='acc_fake')
#plt.xlabel('epoch')
#plt.legend()
#plt.show()
        


#plot gif


#import imageio  
#gif_list = []
#gif_name = 'DCGAN.gif'
#for epoch in range(epochs):
#    image_path = str(epoch)+'.png'
#    image_path = os.path.join(path_gif,image_path)
#    gif_list.append(image_path)
    
    
#def create_gif(gif_list, gif_name):  
#    frames = []  
#    for image_name in gif_list:  
#        frames.append(imageio.imread(image_name))  
#    # Save them as frames into a gif   
#    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.01)  
#  
#    return  
#    
#create_gif(gif_list,gif_name)