#coding:utf-8
#两层简单神经网络（全连接）
import tensorflow as tf
import numpy as np
BATCH SIZE = 8 #定义每次喂入的数据集数
seed = 23455

#基于seed产生随机数
rng = np.random.RandomState(seed)
X= rng.rand(32,2)
Y = [[int(x0 + x1<1)] for (x0,x1) in X]
print ("X:",X)
print ("Y:",Y)

x = tf.placeholder(tf.float32,shape = (None,2))
y_ = tf.placeholder(tf.float32,shape = (None,1))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#定义前向传播过程
a=tf.matmul(x,w1)#矩阵乘法
y=tf.matmul(a,w2)#矩阵乘法
#用会话计算结果
with tf.Session() as sess:
    init_op=tf.global_variables_initializer() #对变量进行初始化
    sess.run(init_op)#填入运算节点
    print ("y in tf3_3.py is:\n",sess.run(y,feed_dict={x:[[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
    print ("w1:",sess.run(w1))
    print ("w2:",sess.run(w2))