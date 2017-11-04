# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
cwd='./data/' 
classes={'class0','class1'}

def main(_):
  for name in enumerate(classes):
    label = name[1][-1]
        # print label
        # print name[1]
    class_path=cwd+name[1]
        # print class_path
    

    for data_path in os.listdir(class_path): 
      data_path=class_path+'/'+data_path #每一个图片的地址
      #print(data_path)

      input = []
      filename = data_path
      for img in os.listdir(filename):
        data = np.genfromtxt(filename+'/'+img, delimiter=',')
        input.append(data[0:200, 1:4])
  

      input[0] = np.expand_dims(input[0], axis=0)
      for i in range(1,4):
        input[i] = np.expand_dims(input[i], axis=0)
        input[0] = np.concatenate((input[0], input[i]), axis=0)
      input[0] = input[0].reshape(1, 2400)
      #print(input[0].shape) # (1, 2400)
      train_data = input[0]

  print(train_data.shape)
  tmp = train_data
  train_data = np.concatenate((tmp, train_data), axis=0)
  print(train_data.shape) # (2, 2400)

  label = np.array(([0, 1], [1, 0]))
  print(label.shape)

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 2400])
  W = tf.Variable(tf.zeros([2400, 2]))
  b = tf.Variable(tf.zeros([2]))
  y = tf.matmul(x, W) + b

  print(mnist.test.images.shape) # numpy.ndarray
  print(mnist.test.labels.shape)
  y_ = tf.placeholder(tf.float32, [None, 2])
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    sess.run(train_step, feed_dict={x: train_data, y_: label})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: train_data,
                                      y_: label}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)