from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=False)
print("finish")
print(mnist)
# data = input.provide_data(mnist)
