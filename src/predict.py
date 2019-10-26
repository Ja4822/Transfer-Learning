import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = '../model/fine_tuned_model'
IMG_SIZE = 64
N_CLASSES = 200

saver = tf.train.import_meta_graph(MODEL_PATH + '/model.ckpt-20.meta')
graph = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
input_images = graph.get_tensor_by_name('input_images:0')# 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
input_labels = graph.get_tensor_by_name('input_labels:0')# 获取输出变量
fc_w = graph.get_tensor_by_name('Logits/fc_2/weights:0')
fc_b = graph.get_tensor_by_name('Logits/fc_2/biases:0')

print(input_images)
print(input_labels)
print(fc_w)
print(fc_b)
