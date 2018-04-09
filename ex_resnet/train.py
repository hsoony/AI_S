import os
import ex_resnet.squeezenet as net

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf

# TensorFlow 내장 MNIST 데이터셋 모듈
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/MNIST", one_hot=True)




inputs = tf.placeholder(tf.float32, [None, 28 * 28])
label = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)  # 배치 정규화 및 dropout 사용시 필요

x = tf.reshape(inputs, shape=[-1, 28, 28, 1])
logit = net.inference(x, is_training)



learning_rate = 0.0001

# 테스트 시 사용하는 연산 그래프 정의
pred_op = tf.nn.softmax(logit)
correct = tf.equal(tf.argmax(label, 1), tf.argmax(pred_op, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())

for step in range(1500):
    batch_X, batch_y = mnist.train.next_batch(32)
    _, loss = sess.run([opt, loss_op], feed_dict={inputs: batch_X, label:batch_y, is_training: True})
    if (step+1)%100 == 0:
        print(step+1, loss)

rst = 0
cnt = 0
for step in range(1500):
    try:
        batch_X, batch_y = mnist.test.next_batch(100)
        acc_test = sess.run(accuracy, feed_dict={inputs:batch_X, label:batch_y, is_training: False})
        # print("test_accuracy:", acc_test )
        rst = rst + float(acc_test)
        cnt = cnt + 1
    except:
        print("end")
print(rst/cnt)