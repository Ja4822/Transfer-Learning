import os
import cv2
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_data import *

MODEL_PATH = '../model/fine_tuned_model'
IMG_PATH = '../data/train/n03085013/images/n03085013_100.JPEG'
CHECKPOINT_PATH = '../model/fine_tuned_model'
IMG_SIZE = 64

img_train_dict, label_dict = create_image_train_dict()
img_val_dict = create_image_val_dict(label_dict)
X_train, y_train, X_test, y_test = create_image_list(img_train_dict, 'train')
X_val, y_val = create_image_list(img_val_dict, 'val')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    variables = tf.trainable_variables()
    for name in variables:
        print(name)

    saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH,  'model.ckpt-51.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
    graph = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
    # 获得输入输出的op
    input_images = graph.get_operation_by_name('input_images').outputs[0]
    img_features = graph.get_operation_by_name('fc_2').outputs
    print(img_features)
    prediction = tf.get_collection('predict')[0]
    result = sess.run(prediction, feed_dict={input_images: X_val[:100]})

    result_idx = sess.run(tf.argmax(result, 1))
    ground_truth_idx = sess.run(tf.argmax(y_val[:100], 1))
    accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(result_idx, ground_truth_idx), dtype=tf.float32)))

    print('[INFO] Prediction is ', result_idx)
    print('[INFO] Ground truth is ', ground_truth_idx)
    print('[INFO] Accuracy is ', accuracy)
