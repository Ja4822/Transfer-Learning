import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
from load_data import *

# tf.enable_eager_execution()

# 保存模型路径
MODEL_PATH = '../model/fine_tuned_model/'
# pre-trained model
CKPT_PATH = '../model/resnet_v1_50.ckpt'
# tensorboard
FILEWRITER_PATH = '../tensorboard/'
# parameters
LEARNING_RATE = 1e-4
EPOCH = 100
BATCH_SIZE = 128
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

    img_train_dict, label_dict = create_image_train_dict()
    img_val_dict = create_image_val_dict(label_dict)
    X_train, y_train, X_test, y_test = create_image_list(img_train_dict, 'train')
    X_val, y_val = create_image_list(img_val_dict, 'val')

    print('====== Seperate Train list and Test list ======')
    print('Train image list: ', np.array(X_train).shape)
    print('Train label list: ', np.array(y_train).shape)
    print(' Test image list: ', np.array(X_test).shape)
    print(' Test label list: ', np.array(y_test).shape)
    print('  Val image list: ', np.array(X_val).shape)
    print('  Val label list: ', np.array(y_val).shape)
    print('================================================')

    input_images = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='input_images')
    input_labels = tf.placeholder(tf.int64, [None, N_CLASSES], name='input_labels')
    # input_images = tf.contrib.eager.Variable(tf.zeros(shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)))
    # input_labels = tf.contrib.eager.Variable(tf.zeros(shape=(BATCH_SIZE, N_CLASSES)))

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # num_classes设置为none时返回softmax之前一层，即图片的向量
        logits, _ = resnet_v1.resnet_v1_50(input_images, num_classes=None, is_training=True)
    
    with tf.variable_scope('Logits'):
        '''
        resnet --> batchnorm1 --> fc1 --> batchnorm2 --> fc2 --> softmax
        '''
        net = tf.squeeze(logits, axis=[1,2])
        batch_norm1 = slim.batch_norm(net, decay=0.9, zero_debias_moving_mean=True, scope='batch_norm1')
        fc_1 = slim.fully_connected(batch_norm1, num_outputs=FC_SIZE,
                weights_initializer=tf.initializers.variance_scaling(), scope='fc_1')
        batch_norm2 = slim.batch_norm(fc_1, decay=0.9, zero_debias_moving_mean=True, scope='batch_norm2')
        fc_2 = slim.fully_connected(batch_norm2, num_outputs=N_CLASSES,
                weights_initializer=tf.initializers.variance_scaling(), scope='fc_2')
        #dropout = slim.dropout(fc_2, scope='dropout')
        logits = slim.softmax(fc_2, scope='softmax')
        tf.add_to_collection("predict", logits)

    trainable_variables = tf.trainable_variables()
    #for name in trainable_variables:
    #    print(name)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # loss
    with tf.variable_scope('loss'):
        
        tf.losses.softmax_cross_entropy(input_labels, logits, weights=1.0)
        loss = tf.losses.get_total_loss()
        #loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=input_labels)
        with tf.control_dependencies(update_ops):
            # 参考AdamOptimizer源码确定参数
            train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, epsilon=1e-4).minimize(loss)

    # acc
    with tf.variable_scope('accuracy'):
        #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_labels, 1))
        predict = tf.argmax(logits, 1)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 加载pre-trained model
    load_fn = slim.assign_from_checkpoint_fn(CKPT_PATH, get_tuned_variables(), ignore_missing_vars=True)

    # 设置batch
    with tf.variable_scope('input_data'):
        train_batch = tf.data.Dataset.from_tensor_slices((input_images, input_labels))
        train_batch = train_batch.shuffle(20).batch(BATCH_SIZE).repeat()
        train_batch_iterator = train_batch.make_initializable_iterator()
        train_batch_data = train_batch_iterator.get_next()
        val_batch = tf.data.Dataset.from_tensor_slices((input_images, input_labels))
        val_batch = val_batch.shuffle(20).batch(BATCH_SIZE).repeat()
        val_batch_iterator = val_batch.make_initializable_iterator()
        val_batch_data = val_batch_iterator.get_next()

    # tensorboard
    with tf.variable_scope('summary'):
        # acc, loss曲线
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        # 新添加层的输出
        tf.summary.histogram('bn1', batch_norm1)
        tf.summary.histogram('fc_1', fc_1)
        tf.summary.histogram('bn2', batch_norm2)
        tf.summary.histogram('fc_2', fc_2)
        tf.summary.histogram('softmax_output', logits)
        #　合并
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FILEWRITER_PATH + '/train')
        saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # 初始化
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        print('====== LOAD PRE-TRAINED MODEL ======')
        load_fn(sess)
        print('========== START TRAINING ==========')
        num_iteration = np.array(X_train).shape[0] // BATCH_SIZE + 1
        # 初始化迭代器
        sess.run(train_batch_iterator.initializer, 
                feed_dict={input_images: X_train, input_labels: y_train})
        sess.run(val_batch_iterator.initializer, 
                feed_dict={input_images: X_test, input_labels: y_test})
        # sess.run(train_batch_iterator.initializer)
        # sess.run(val_batch_iterator.initializer)
        
        writer.add_graph(sess.graph)
        # 每个epoch
        num = 1
        for epoch in range(EPOCH):
            # train
            train_loss_mean = []
            train_acc_mean = []
            for train_batch_count in range(num_iteration):
                X_train_batch, y_train_batch = sess.run(train_batch_data)
                train_loss, train_acc, _ = sess.run([loss, accuracy, train_op],
                            feed_dict={input_images: X_train_batch, input_labels: y_train_batch})
                # plot_images_list(X_train_batch, y_train_batch)
                train_loss_mean.append(train_loss)
                train_acc_mean.append(train_acc)
                if train_batch_count % 10 == 0:
                    train_predict, summary = sess.run([predict, merged_summary],
                            feed_dict={input_images:X_train_batch, input_labels:y_train_batch})

                    print('EPOCH %d/%d BATCH %d/%d, TRAIN LOSS=%.3f, TRAIN ACC=%.3f'\
                            %(epoch+1, EPOCH, train_batch_count+1, num_iteration, train_loss, train_acc))

                    writer.add_summary(summary, num)
                    num += 1
            # 每次epoch对所有val进行验证
            val_acc_mean = []
            val_loss_mean = []
            for val_batch_count in range(np.array(X_test).shape[0] // BATCH_SIZE + 1):
                X_test_batch, y_test_batch = sess.run(val_batch_data)
                val_loss, val_acc, val_prediction, _ = sess.run([loss, accuracy, predict, train_op], 
                                feed_dict={input_images: X_test_batch, input_labels: y_test_batch})
                val_acc_mean.append(val_acc)
                val_loss_mean.append(val_loss)
            print('EPOCH %d TRAIN LOSS=%.3f, TRAIN ACC=%.3f, VAL LOSS=%.3f, VAL ACC=%.3f'\
                %(epoch+1, np.mean(train_loss_mean), np.mean(train_acc_mean), np.mean(val_loss_mean), np.mean(val_acc_mean)))
            
            # 每10个epoch保存一次模型
            if epoch % 50 == 0 or epoch + 1 == EPOCH:
                saver.save(sess, MODEL_PATH+'model.ckpt', epoch+1)
                print('EPOCH %d, Save checkpoints in %s'%(epoch+1, MODEL_PATH+'model.ckpt-'+str(epoch+1)))
            print('================================================')

        print('====== FINISH TRAINING ======')
        test_acc, test_prediction, _ = sess.run([accuracy, predict, train_op], feed_dict={input_images: X_val[:100], input_labels: y_val[:100]})
        print('test acc = %.3f'%(test_acc))
        plot_images_list(X_val[:100], y_val[:100], test_prediction)
        
if __name__ == '__main__':
    main()