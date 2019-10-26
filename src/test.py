import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1

N_CLASSES = 200
IMG_SIZE = 64

input_images = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='input_images')
input_labels = tf.placeholder(tf.int64, [None, N_CLASSES], name='input_labels')

with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits, _ = resnet_v1.resnet_v1_50(input_images, num_classes=N_CLASSES, is_training=True)
    print(logits)