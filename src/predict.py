import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = '../model/fine_tuned_model'
IMG_PATH = '../data/train/n01443537/images/n01443537_10.JPEG'
CHECKPOINT_PATH = '../model/fine_tuned_model'
IMG_SIZE = 64

img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
img = img / 255.0
img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
#img = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH,  'model.ckpt-50.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
    graph = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
    # 获得输入输出的op
    input_images = graph.get_operation_by_name('input_images').outputs[0]
    prediction = tf.get_collection('predict')[0]
    result = sess.run(prediction, feed_dict={input_images: img})
    result = np.argmax(result)
    print('[INFO] Prediction is %d'%(result))
