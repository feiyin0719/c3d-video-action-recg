# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_fine_test
   Description :
   Author :       iffly
   date：          5/9/18
-------------------------------------------------
   Change Activity:
                   5/9/18:
-------------------------------------------------
"""
import tensorflow as tf
from tensorflow.contrib import slim

from config import Config
from dataset_records import dataset_records
from model import C3d
import numpy as np
if __name__=='__main__':
    with tf.Graph().as_default():

        # exclude = ['var_name/wd1', 'var_name/wd2','var_name/bd1','var_name/bd2','var_name/wout','var_name/bout']
        # variables_to_restore = slim.get_variables_to_restore(exclude=exclude)


        images_placeholder = tf.placeholder(tf.float32, shape=(1,
                                                               Config.channels,
                                                               Config.image_size,
                                                               Config.image_size,
                                                               3))
        keep_prob = tf.placeholder(tf.float32)
        net=C3d(num_class=20,keep_prob=keep_prob).build_model(images_placeholder)

        sess=tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver=tf.train.Saver()
        saver.restore(sess,'/home/iffly/dataset/finetuning_ucf101.model')
        data_set = dataset_records('/home/iffly/dataset/train_data.tfrecords', batch_size=1, epoch=1)
        iterator = data_set.make_one_shot_iterator()
        one_element = iterator.get_next()
        data=sess.run(one_element)
        label=sess.run(net,feed_dict={images_placeholder:data[0]/255,keep_prob:1})
        print label
        label=np.exp(label)/np.sum(np.exp(label))
        print label,np.argmax(label)