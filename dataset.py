# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     dataset
   Description :
   Author :       iffly
   date：          5/7/18
-------------------------------------------------
   Change Activity:
                   5/7/18:
-------------------------------------------------
"""
import cPickle as pickle
import os
import tensorflow as tf

from config import Config


def parse_function(files, label):
    images = []
    print files
    for i in range(0, 2):
        file = tf.slice(files, [0, 0], [-1, 1])
        image_string = tf.read_file(file)
        image_decoded = tf.image.decode_image(image_string)
        image_resized = tf.image.resize_images(image_decoded, [Config.image_size, Config.image_size])
        images.append(image_resized)
    images = tf.stack(images, axis=1)
    return images, label


def get_dataset(features, labels, epoch=10, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(reshuffle_each_iteration=True, buffer_size=len(labels)).batch(
        batch_size=batch_size).repeat(epoch)
    return dataset


if __name__ == '__main__':
    with open('./train_data.pickle', 'r') as f:
        data = pickle.load(f)
    features = data['data']
    labels = data['label']
    data_set = get_dataset(features, labels, batch_size=1)
    iterator = data_set.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")
