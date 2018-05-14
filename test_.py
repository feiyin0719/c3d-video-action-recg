# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       iffly
   date：          5/7/18
-------------------------------------------------
   Change Activity:
                   5/7/18:
-------------------------------------------------
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from model import C3d

if __name__ == '__main__':
    # image_names=['./images/frame/cut/cut_s7/0.jpg','./images/frame/cut/cut_s7/10.jpg']
    # # 输出TFRecord文件的地址
    # filename = "./mnist_output.tfrecords"
    # # 创建一个writer来写TFRecord文件
    # writer = tf.python_io.TFRecordWriter(filename)
    # images=[]
    # images.append(cv2.imread(image_names[0]))
    # images.append(cv2.imread(image_names[1]))
    # images=np.array(images)
    # print images
    # example = tf.train.Example(features=tf.train.Features(feature={
    #     'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images.tostring()]))}))
    # writer.write(example.SerializeToString())
    # writer.close()
    # # 创建一个reader来读取TFRecord文件中的样例
    # reader = tf.TFRecordReader()
    # # 创建一个队列来维护输入文件列表
    # filename_queue = tf.train.string_input_producer([filename])
    #
    # # 从文件中读出一个样例，也可以使用read_up_to函数一次性读取多个样例
    # # _, serialized_example = reader.read(filename_queue)
    # _, serialized_example = reader.read_up_to(filename_queue, 1)  # 读取6个样例
    # # 解析读入的一个样例，如果需要解析多个样例，可以用parse_example函数
    # # features = tf.parse_single_example(serialized_example, features={
    # # 解析多个样例
    # features = tf.parse_example(serialized_example, features={
    #     # TensorFlow提供两种不同的属性解析方法
    #     # 第一种是tf.FixedLenFeature,得到的解析结果为Tensor
    #     # 第二种是tf.VarLenFeature,得到的解析结果为SparseTensor，用于处理稀疏数据
    #     # 解析数据的格式需要与写入数据的格式一致
    #     'image_raw': tf.FixedLenFeature([], tf.string),
    # })
    #
    # # tf.decode_raw可以将字符串解析成图像对应的像素数组
    # images = tf.decode_raw(features['image_raw'], tf.uint8)
    # sess = tf.Session()
    # # 启动多线程处理输入数据
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # image = sess.run([images])
    # print image[0].reshape(2,1920,1080,3)
    # sess.close()
    print ("{0}:train accuracy: {1:.5f}".format(2, 0.5))
