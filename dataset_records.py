# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     dataset_records
   Description :
   Author :       iffly
   date：          5/8/18
-------------------------------------------------
   Change Activity:
                   5/8/18:
-------------------------------------------------
"""
import cv2
import tensorflow as tf

from config import Config
import numpy as np
np_mean = np.load('crop_mean.npy').reshape([Config.channels, Config.image_size, Config.image_size, 3])
def parse_exmp(serial_exmp):
    feats = tf.parse_single_example(serial_exmp, features={'images':tf.FixedLenFeature([], tf.string),
        'label':tf.FixedLenFeature([],tf.int64)})
    images = tf.decode_raw(feats['images'], tf.uint8)
    images=tf.cast(images,tf.float32)
    images=tf.reshape(images,shape=[Config.channels,Config.image_size,Config.image_size,3])
    images=tf.subtract(images,np_mean)
    # label =tf.one_hot (feats['label'],20,1,0)
    label=feats['label']
    return images, label
def dataset_records(record_path,batch_size=4,epoch=10):
    dataset = tf.data.TFRecordDataset(record_path)
    dataset=dataset.map(parse_exmp).shuffle(reshuffle_each_iteration=True,buffer_size=1000).batch(batch_size=batch_size).repeat(epoch)
    return dataset
if __name__ =='__main__':
    data_set = dataset_records('/home/iffly/dataset/train_data.tfrecords', batch_size=1,epoch=1)
    iterator = data_set.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                data=sess.run(one_element)
                print(data[0])
                cv2.imwrite("test.jpg",data[0][0][0])
        except tf.errors.OutOfRangeError:
            print("end!")