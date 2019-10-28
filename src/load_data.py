import glob
import os
import cv2
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

TEST_PERCENTAGE = 0.1
VALIDATION_PERCENTAGE = 0.1
IMG_SIZE = 64
N_CLASSES = 200

# 读取图片信息
def create_image_train_dict():
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
            if i == 201:
                break
            img_path.append(os.path.join(label_path, img_dir_name))
            j += 1
        image_dict[label_name] = img_path
        label_dict[label_name] = []
    labels = list(image_dict.keys())
    print('label num = ', N_CLASSES)
    print('Num of images: %d'%(j))
    print(label_dict)
    # 将label的序号的对应存入文件，以便预测时对应
    # labels.txt与img_train_dict中的key顺序相同
    with open(LABLES_PATH, 'w') as f:
        for i in range(len(labels)):
            line = str(i)+','+labels[i]+'\n'
            f.write(line)
    print('Save labels in %s'%(LABLES_PATH))
    return image_dict, label_dict

def create_image_val_dict(label_dict):
    names = []
    labels = []
    with open(os.path.join(VAL_DATA_PATH, 'val_annotations.txt'), 'r') as f:
        for line in f.readlines():
            data = line.split('\t')
            names.append(data[0])
            labels.append(data[1])
    label_dict_labels = list(label_dict.keys())
    for i in range(len(labels)):
        idx = label_dict_labels.index(labels[i])
        img_path = os.path.join(VAL_DATA_PATH, os.path.join('images', names[i]))
        label_dict[label_dict_labels[idx]].append(img_path)
    '''
    for i in range(len(names)):
        img_val_dict[labels[i]] = []
    dict_keys = list(img_val_dict.keys())
    for i in range(len(names)):
        index = dict_keys.index(labels[i])
        img_path = os.path.join(VAL_DATA_PATH, os.path.join('images', names[i]))
        img_val_dict[dict_keys[index]].append(img_path)
    '''

    return label_dict

# 打乱数据集并划分
def create_image_list(img_dict, category):
    label_list = []
    img_list = []
    labels = list(img_dict.keys())
    for label, img_path in img_dict.items():
        # 将label转为on-hot编码
        tmp = np.zeros(N_CLASSES, dtype=np.float32)
        tmp[labels.index(label)] = 1.0
        #print(tmp)
        # 保存img和label
        for i in range(len(img_path)):
            img = cv2.cvtColor(cv2.imread(img_path[i]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            img = img / 255.0
            img_list.append(img)
            label_list.append(tmp)
        print('Finish %s'%(label))

    if category == 'train':
        X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(
            img_list, label_list, test_size=TEST_PERCENTAGE, random_state=np.random.randint(0,1000))
        print('[INFO] Finish loading train data')
        with open('../model/train.txt', 'w') as f:
            for i in range(len(labels)):
                line = str(i)+','+labels[i]+'\n'
                f.write(line)
        return X_train_list, y_train_list, X_test_list, y_test_list

    if category == 'val':
        X_val_list, _, y_val_list, _ = train_test_split(
            img_list, label_list, test_size=0.0, random_state=np.random.randint(0,1000))
        with open('../model/val.txt', 'w') as f:
            for i in range(len(labels)):
                line = str(i)+','+labels[i]+'\n'
                f.write(line)
        print('[INFO] Finish loading val data')

        return X_val_list, y_val_list
def get_images(X_list):
    img_list = []
    for img_path in X_list:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        img = img / 255.0
        img_list.append(img)
    return img_list

# 显示图片
def plot_images_list(img_list, label_list, prediction=[]):
    if prediction == []:
        fig = plt.gcf()
        fig.set_size_inches(12,14)
        for i in range(0, 25):
            ax = plt.subplot(5,5,i+1)
            label = np.argmax(label_list[i])
            #print(img_list[i])
            ax.imshow(img_list[i])
            ax.set_title(label, fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
        plt.show()
    else:
        fig = plt.gcf()
        fig.set_size_inches(12,14)
        for i in range(0, 25):
            ax = plt.subplot(5,5,i+1)
            label = np.argmax(label_list[i])
            #print(img_list[i])
            ax.imshow(img_list[i])
            ax.set_title('truth:'+str(label)+',predict:'+str(prediction[i]), fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
        plt.show()

def get_next_batch(X_train, y_train, i, batch_size):
    X_train_batch = []
    y_train_batch = []
    if (i+1)*batch_size<len(X_train):
        n = (i+1)*batch_size
    else:
        n = len(X_train)
    for i in range(i*batch_size, n):
        X_train_batch.append(X_train[i])
        y_train_batch.append(y_train[i])
    return X_train_batch, y_train_batch

def get_val_batch(X_val, y_val, batch_size):
    X_list = []
    y_list = []
    for _ in range(batch_size):
        idx = np.random.randint(0,len(X_val))
        X_list.append(X_val[idx])
        y_list.append(y_val[idx])
    return X_list, y_list

# def main():
#     img_train_dict = create_image_train_dict()
#     img_val_dict = create_image_val_dict()
#     plot_images_labels(img_train_dict)
#     plot_images_labels(img_val_dict)
#     X_train, y_train, X_test, y_test = create_image_list(img_train_dict, 'train')
#     X_val, y_val = create_image_list(img_val_dict, 'val')

# if __name__ == "__main__":
#     main()