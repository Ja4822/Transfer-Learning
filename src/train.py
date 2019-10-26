import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
from load_data import *

# 保存模型路径
MODEL_PATH = '../model/fine_tuned_model/'
# pre-trained model
CKPT_PATH = '../model/resnet_v1_50.ckpt'
# tensorboard
FILEWRITER_PATH = '../tensorboard/'
# parameters
LEARNING_RATE = 1e-3
EPOCH = 20
BATCH_SIZE = 256
N_CLASSES = 200
IMG_SIZE = 64
FC_SIZE = 2048

# 不需要从训练好的模型中加载的参数，就是最后的自定义的全连接层
CHECKPOINT_EXCLUDE_SCOPES = 'Logits'
# 指定最后的全连接层为可训练的参数
TRAINABLE_SCOPES = 'Logits'

# 加载所有需要从训练好的模型加载的参数
def get_tuned_variables():
    ##不需要加载的范围
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(",")]
    # 初始化需要加载的参数
    variables_to_restore = []

    # 遍历模型中的所有参数
    for var in slim.get_model_variables():
        # 先指定为不需要移除
        excluded = False
        # 遍历exclusions，如果在exclusions中，就指定为需要移除
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        # 如果遍历完后还是不需要移除，就把参数加到列表里
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

# 获取所有需要训练的参数
def get_trainable_variables():
    # 同上
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(",")]
    variables_to_train = []
    # 枚举所有需要训练的参数的前缀，并找到这些前缀的所有参数
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def main():
    img_train_dict = create_image_train_dict()
    img_val_dict = create_image_val_dict()
    plot_images_labels(img_train_dict)
    plot_images_labels(img_val_dict)
    X_train, y_train, X_test, y_test = create_image_list(img_train_dict, 'train')
    X_val, y_val = create_image_list(img_val_dict, 'val')

    print('====== Seperate Train list and Test list ======')
    print('Train image list: ', np.array(X_train).shape)
    print('Train label list: ', np.array(y_train).shape)
    print('Test image list: ', np.array(X_test).shape)
    print('Test label list: ', np.array(y_test).shape)
    print('Val image list: ', np.array(X_val).shape)
    print('Val label list: ', np.array(y_val).shape)
    print('================================================')

    input_images = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='input_images')
    input_labels = tf.placeholder(tf.int64, [None, N_CLASSES], name='input_labels')

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # num_classes设置为none时返回softmax之前一层，即图片的向量
        logits, _ = resnet_v1.resnet_v1_50(input_images, num_classes=None, is_training=True)
    
    with tf.variable_scope('Logits'):
        net = tf.squeeze(logits, axis=[1,2])
        #net = slim.dropout(net, keep_prob=0.5, scope='dropout_scope')
        #batch_norm1 = slim.batch_norm(net, scope='batch_norm1')
        fc_1 = slim.fully_connected(net, num_outputs=FC_SIZE, 
                weights_initializer=tf.initializers.variance_scaling(), scope='fc_1')
        #batch_norm2 = slim.batch_norm(fc_1, scope='batch_norm2')
        fc_2 = slim.fully_connected(fc_1, num_outputs=N_CLASSES, 
                weights_initializer=tf.initializers.variance_scaling(), scope='fc_2')
        logits = slim.softmax(fc_2, scope='softmax')
    
    trainable_variables = tf.trainable_variables()

    #for name in trainable_variables:
    #    print(name)
    # loss
    with tf.variable_scope('loss'):
        #tf.losses.softmax_cross_entropy(input_labels, logits, weights=1.0)
        #loss = tf.losses.get_total_loss()
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=input_labels)
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # acc
    with tf.variable_scope('accuracy'):
        #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_labels, 1))
        predict = tf.argmax(logits, 1)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 加载pre-trained model
    load_fn = slim.assign_from_checkpoint_fn(CKPT_PATH, get_tuned_variables(), ignore_missing_vars=True)

    # tensorboard
    with tf.variable_scope('summary'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FILEWRITER_PATH + '/train')
        valid_writer = tf.summary.FileWriter(FILEWRITER_PATH + '/valid')
        saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('====== LOAD PRE-TRAINED MODEL ======')
        load_fn(sess)
        print('========== START TRAINING ==========')
        num_iteration = np.array(X_train).shape[0] // BATCH_SIZE + 1

        # 每个epoch
        num = 1
        for epoch in range(EPOCH):
            
            train_writer.add_graph(sess.graph)
            
            for train_batch_count in range(num_iteration):
                X_train_batch, y_train_batch = get_next_batch(X_train, y_train, train_batch_count, BATCH_SIZE)
                sess.run(train_op, 
                            feed_dict={input_images: X_train_batch, input_labels: y_train_batch})

                if train_batch_count % 10 == 0:
                    train_loss, train_acc = sess.run([loss, accuracy],
                            feed_dict={input_images: X_train_batch, input_labels: y_train_batch})
                    train_predict = sess.run(predict, feed_dict={input_images:X_train_batch, input_labels:y_train_batch})
                    print(train_predict[:50])
                    print(np.argmax(y_train_batch[:50], 1))
                    print('EPOCH %d/%d BATCH %d/%d, TRAIN LOSS=%.3f, TRAIN ACC=%.3f'\
                            %(epoch+1, EPOCH, train_batch_count+1, num_iteration, train_loss, train_acc))

                    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    #run_metadata = tf.RunMetadata()
                    summary, _ = sess.run(
                        [merged_summary, train_op], 
                        feed_dict={input_images: X_train_batch, input_labels: y_train_batch})
                        #options=run_options,
                        #run_metadata=run_metadata)
                    #rain_writer.add_run_metadata(run_metadata, 'Epoch %03d Batch %03d' %(train_batch_count, epoch))
                    train_writer.add_summary(summary, num)
                    #print('Adding run metadata for', num)
                    num += 1
                
            #　每次epoch进行验证
            X_test_batch, y_test_batch = get_val_batch(X_test, y_test, BATCH_SIZE)
            val_loss, val_acc, _ = sess.run([loss, accuracy, train_op], 
                            feed_dict={input_images: X_test_batch, input_labels: y_test_batch})
            print('EPOCH %d VAL LOSS=%.3f VAL ACC=%.3f'%(epoch+1, val_loss, val_acc))
            print('================================================')
            if epoch % 10 == 0 or epoch + 1 == EPOCH:
                saver.save(sess, MODEL_PATH+'model.ckpt', epoch+1)
                print('EPOCH %d, Save checkpoints in %s'%(epoch+1, MODEL_PATH+'model.ckpt-'+str(epoch+1)))

        print('====== FINISH TRAINING ======')
        test_acc, prediction = sess.run([accuracy, predict], feed_dict={input_images: X_val[:10], input_labels: y_val[:10]})
        print('test acc = %.3f'%(test_acc))
        print('ground_truth: ', np.argmax(np.array(y_val), 1))
        print('predict: ', prediction)
        train_writer.close()

if __name__ == '__main__':
    main()