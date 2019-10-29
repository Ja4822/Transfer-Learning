import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from collections import OrderedDict


TRAIN_DATA_PATH = '../data/train'
VAL_DATA_PATH = '../data/val'
LABLES_PATH = '../model/labels.txt'
MODEL_PATH = '../model/fine_tuned_model'
IMG_PATH = '../data/train/n03085013/images/n03085013_100.JPEG'
CHECKPOINT_PATH = '../model/fine_tuned_model'

TEST_PERCENTAGE = 0.1
VALIDATION_PERCENTAGE = 0.1
N_CLASSES = 200
IMG_SIZE = 64
BATCH_SIZE = 128

# 读取图片信息,num为每个类别的数量
def create_image_train_dict(num=200):
    image_dict = OrderedDict()
    label_dict = OrderedDict()
    j = 0
    sub_dir_names = os.listdir(TRAIN_DATA_PATH)
    for sub_dir_name in sub_dir_names:
        label_path = os.path.join(TRAIN_DATA_PATH, sub_dir_name)
        label_path = os.path.join(label_path, 'images')
        label_name = sub_dir_name
        img_dir_names = os.listdir(label_path)
        img_path = []
        i = 0
        for img_dir_name in img_dir_names:
            i += 1
            if i == num+1:
                break
            img_path.append(os.path.join(label_path, img_dir_name))
            j += 1
        image_dict[label_name] = img_path
        label_dict[label_name] = []
    labels = list(image_dict.keys())
    print('label num = ', N_CLASSES)
    print('Num of images: %d'%(j))
    # 将label的序号的对应存入文件，以便预测时对应
    # labels.txt与img_train_dict中的key顺序相同
    return image_dict, label_dict

# 打乱数据集并划分
def create_image_list(img_dict, category):
    label_list = []
    img_list = []
    labels = list(img_dict.keys())
    #print('[INFO] %s labels: '%(category), labels)
    print('=================================')
    for idx, label in enumerate(labels):
        img_path = img_dict[label]
        for i in range(len(img_path)):
            img = cv2.cvtColor(cv2.imread(img_path[i]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            img = img / 255.0
            img_list.append(img)
            label_list.append(idx)
        #print('Finish %s'%(label))
    # 将label转为on-hot编码
    print('[INFO] Convert labels to one-hot encoder')
    with tf.Session() as sess:
        label_list = sess.run(tf.one_hot(label_list, N_CLASSES, 1, 0))
    return img_list, label_list

def get_cos_distance(sess, x1, x2):
    # x1 = tf.reshape(x1, [1, 2048])
    # x2 = tf.reshape(x2, [1, 2018])
    x1 = tf.expand_dims(x1, 0)
    x2 = tf.expand_dims(x2, 0)
    x1_x2 = tf.matmul(x1, tf.transpose(x2))
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), 1, keepdims=True))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), 1, keepdims=True))
    # print(x1_x2, x1_norm, x2_norm)
    return sess.run(x1_x2 / (x1_norm*x2_norm))


img_train_dict, label_dict = create_image_train_dict(num=200)
X_train, y_train = create_image_list(img_train_dict, 'train')

print('=================================================')
with tf.Session() as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())

    # 加载模型
    saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH,  'model.ckpt-100.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
    graph = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
    
    # 获得输入输出的op
    input_images = graph.get_operation_by_name('input_images').outputs[0]
    input_labels = graph.get_operation_by_name('input_labels').outputs[0]
    img_features = graph.get_tensor_by_name('Logits/softmax/Reshape:0')
    prediction = tf.get_collection('predict')[0]

    print(img_features)
    print(prediction)
    
    print('========== CALCULATE COS SIMILARITY ==========')
    # 计算类别内的cos_sim
    cos_sim_list = []
    for i in range(N_CLASSES):
        print('class %d'%(i))
        # 计算当前类别的向量
        num = 0.0
        cos_sim = 0.0
        img_vector = sess.run(img_features, feed_dict={input_images: X_train[i*200+100:(i+1)*200+100]})
        for j in range(int(len(img_vector)/2)):
            for k in range(int(len(img_vector)/2), len(img_vector)):
                tmp = get_cos_distance(sess, img_vector[j], img_vector[k])
                print(tmp)
                cos_sim += get_cos_distance(sess, img_vector[j], img_vector[k])
                num += 1.0
        cos_sim_mean = cos_sim / num
        cos_sim_list.append(cos_sim_mean)
    print(cos_sim_list)
    print('===============================================')
    # test_batch = tf.data.Dataset.from_tensor_slices((input_images, input_labels))
    # test_batch = test_batch.shuffle(20).batch(BATCH_SIZE).repeat()
    # test_batch_iterator = test_batch.make_initializable_iterator()
    # test_batch_data = test_batch_iterator.get_next()

    # sess.run(test_batch_iterator.initializer, 
    #             feed_dict={input_images: X_train, input_labels: y_train})
    acc_list = []
    total_num = np.array(X_train).shape[0]
    num_iteration = total_num // BATCH_SIZE + 1
    for i in range(num_iteration):
        print('%d/%d'%((i+1), num_iteration))
        # X_test_batch, y_test_batch = sess.run(test_batch_data)
        if (i+1)*BATCH_SIZE < total_num:
            X_test_batch = X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_test_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        else:
            X_test_batch = X_train[i*BATCH_SIZE:total_num]
            y_test_batch = y_train[i*BATCH_SIZE:total_num]

        result = sess.run(prediction, feed_dict={input_images: X_test_batch})
        

        result_idx = sess.run(tf.argmax(result, 1))
        ground_truth_idx = sess.run(tf.argmax(y_test_batch, 1))
        accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(result_idx, ground_truth_idx), dtype=tf.float32)))
        acc_list.append(accuracy)
    acc_mean = np.mean(np.array(acc_list))

    print('[INFO] Prediction is ', result_idx)
    print('[INFO] Ground truth is ', ground_truth_idx)
    print('[INFO] Accuracy is ', acc_mean)
