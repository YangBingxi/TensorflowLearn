#coding:utf-8
import tensorflow as tf #引入模块
x = tf.constant([[1.0, 2.0]]) #定义一个 2 阶张量等于[[1.0,2.0]]
w = tf.constant([[3.0], [4.0]]) #定义一个 2 阶张量等于[[3.0],[4.0]]
y = tf.matmul(x, w) #实现 xw 矩阵乘法
print (y) #打印出结果
with tf.Session() as sess:
print (sess.run(y))
#执行会话并打印出执行后的结果
