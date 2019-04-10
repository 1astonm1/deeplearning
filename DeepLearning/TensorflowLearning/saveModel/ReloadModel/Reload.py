import tensorflow as tf
import matplotlib as plt
import numpy as np


def reload_model():
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('./model/MyModel!.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./model'))


if __name__ == '__main__':
    reload_model()

