# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 02:38:32 2017

@author: preetish
"""
#image classifier on mnist and cifar10 using CNN(Lenet5) 

import tensorflow as tf
import numpy as np 
import cv2
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

Mnist=input_data.read_data_sets('/data/mnist',one_hot=True)
'''
LeNet-5

parameters
f1(5x5x6)
f2(5x5x16)
maxpool(2x2),stride=2
fc 128,84

'''
learningrate=0.001
epochs=10
batchsize=1000


def Dense(X,W,b):
    return tf.nn.relu(tf.nn.bias_add(tf.matmul(X,W),b))

#parameters
w1=tf.Variable(initial_value=tf.random_normal(shape=[5,5,3,12]))
w2=tf.Variable(initial_value=tf.random_normal([5,5,12,32]))
wd1=tf.Variable(initial_value=tf.random_normal([2048,256]))
wd2=tf.Variable(initial_value=tf.random_normal([256,168])) 
wy=tf.Variable(initial_value=tf.random_normal([84,10])) 
b1=tf.Variable(initial_value=tf.zeros([12]))
b2=tf.Variable(initial_value=tf.zeros([32]))
bd1=tf.Variable(initial_value=tf.zeros([256]))
bd2=tf.Variable(initial_value=tf.zeros([168]))
by=tf.Variable(initial_value=tf.zeros([10]))

#placeholders
X=tf.placeholder(tf.float32,(None,32,32,3),name='X')
y=tf.placeholder(tf.float32,(None,10),name='y') 
keep_ratio = tf.placeholder(tf.float32)

#model
#layer1
dX= tf.reshape(X, shape=[-1,32,32,3])
filter1=tf.nn.bias_add(tf.nn.conv2d(dX,w1,strides=[1,1,1,1],padding='SAME'),b1)
layer1=tf.nn.relu(filter1)
layer1=tf.nn.max_pool(layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#layer2
filter2=tf.nn.bias_add(tf.nn.conv2d(layer1,w2,strides=[1,1,1,1],padding='SAME'),b2)
layer2=tf.nn.relu(filter2)
layer2=tf.nn.max_pool(layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#flatten           
layer3=tf.reshape(layer2,[-1,int(layer2.get_shape().as_list()[1])*int(layer2.get_shape().as_list()[2])*int(layer2.get_shape().as_list()[3])])
#dense
dense1=Dense(layer3,wd1,bd1)
drop1=tf.nn.dropout(dense1,keep_ratio)
dense2=Dense(drop1,wd2,bd2)
drop2=tf.nn.dropout(dense2,keep_ratio)
dense3=tf.nn.bias_add(tf.matmul(drop2,wy),by)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense3,labels=y))
output=tf.nn.softmax(dense3)

grad=tf.train.AdamOptimizer(learning_rate=learningrate).minimize(cost)

'''load data for Cifar10'''

tx=[]
ty=[]
tex=[]
tey=[]

labels=pd.read_csv('C:/Users/preetish/Downloads/train/trainLabels.csv')
dy=pd.get_dummies(labels)
dy=dy.drop('id',axis=1)
ty=np.array(dy)

  
for i in np.arange(1,50001):
    file1='C:/Users/preetish/Downloads/train/'+str(i)+'.png'
    f1=cv2.imread(file1)
    f1 = cv2.resize(f1, (32,32), cv2.INTER_LINEAR)
    tex.append(f1)

tex = np.array(tex, dtype=np.uint8)

tex= tex.astype('float32')

tex=np.multiply(tex,1.0/255.0)

'''Cifar10'''
with tf.Session() as Sess:
    
    Sess.run(tf.global_variables_initializer())
    
    for i in range(10):
        for j in range(int(50000/batchsize)):
            X_batch,Y_batch=tex[j*batchsize:(j+1)*batchsize],ty[j*batchsize:(j+1)*batchsize]
            Sess.run(grad,feed_dict={X:X_batch,y:Y_batch,keep_ratio:0.7})
     
    predict=Sess.run(output,feed_dict={X:tex,y:ty,keep_ratio:1.0})
    correct=tf.equal(tf.arg_max(predict,1),tf.arg_max(ty,1))
    accuracy=tf.reduce_sum(tf.cast(correct,tf.float32))
    total_corr=Sess.run(accuracy)
    acc=total_corr/50000
    print(acc)
 

'''mnist'''#got 0.97 train accuracy in 10 epochs using lenet5
#while using mnist change the model parameters to 28X28X1 

with tf.Session() as Sess:
    Sess.run(tf.global_variables_initializer())
    for i in range(epochs):
            for j in range(int(Mnist.train.num_examples/batch_size)):
                X_batch,Y_batch=Mnist.train.next_batch(batch_size)
                Sess.run(grad,feed_dict={X:X_batch,y:Y_batch,keep_ratio:0.7})
    
    for j in range(int(Mnist.test.num_examples/batch_size)):
            X_test,Y_test=Mnist.test.next_batch(batch_size)
            predict=Sess.run(output,feed_dict={X:X_test,keep_ratio:1.0})
            correct=tf.equal(tf.arg_max(predict,1),tf.arg_max(Y_test,1))
            accuracy=tf.reduce_sum(tf.cast(correct,tf.float32))
            total_corr+=Sess.run(accuracy)
    acc=total_corr/Mnist.test.num_examples
    print(acc)
    
