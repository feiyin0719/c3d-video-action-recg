# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train
   Description :
   Author :       iffly
   date：          5/12/18
-------------------------------------------------
   Change Activity:
                   5/12/18:
-------------------------------------------------
"""
import os
import tensorflow as tf

import dataset_records
from config import Config
from model import C3d

BATCH_SIZE = 4
# if not finetune let FINE_PATH=""
FINE_PATH = '/home/iffly/dataset/finetuning_ucf101.model'
TRAIN_PATH = '/home/iffly/dataset/train_data.tfrecords'
TEST_PATH = '/home/iffly/dataset/train_data.tfrecords'
EPOCH = 10
NUM_CLASS = 20
MOVING_AVERAGE_DECAY = 0.9999
MODEL_SAVE_PATH = '/home/iffly/dataset/models'
MODEL_BEST_PATH = '/home/iffly/dataset/best'
TENSORBOARD_PATH = '/home/iffly/dataset/visual_logs/'
# not finetune use this
LR1 = 1e-4
LR2 = 1e-4
# finetune use this
# LR1=1e-5
# LR2=1e-4
TRAIN_LEN = 2000
DECAY_RATE = 0.9
DECAY_RATE1 = 0.9
STEP_INV1 = 5
STEP_INV2 = int(TRAIN_LEN/BATCH_SIZE/2)  # =TRAIN_LEN/2



def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
if not os.path.exists(MODEL_BEST_PATH):
    os.makedirs(MODEL_BEST_PATH)
with tf.Graph().as_default():
    global_step = tf.get_variable(
        'global_step',
        [],
        initializer=tf.constant_initializer(0),
        trainable=False, type=tf.int32
    )

    global_step1 = tf.get_variable(
        'global_step1',
        [],
        initializer=tf.constant_initializer(0),
        trainable=False, type=tf.int32
    )
    # if fine tune use the code
    # learning_rate=tf.train.exponential_decay(LR1,global_step=global_step,decay_rate=DECAY_RATE,decay_steps=TRAIN_LEN/BATCH_SIZE,staircase=True)
    # learning_rate1 = tf.train.exponential_decay(LR2, global_step=global_step1, decay_rate=DECAY_RATE1,
    #                                            decay_steps=TRAIN_LEN / BATCH_SIZE, staircase=True)
    # not fine tune use the code
    boundaries1 = [int(TRAIN_LEN / BATCH_SIZE * 5), int(TRAIN_LEN / BATCH_SIZE * 7)]
    values1 = [LR1, LR1 * 0.1, LR1 * 0.01]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries1, values1)
    boundaries2 = [int(TRAIN_LEN / BATCH_SIZE * 5), int(TRAIN_LEN / BATCH_SIZE * 7)]
    values2 = [LR2, LR2 * 0.1, LR2 * 0.01]
    learning_rate1 = tf.train.piecewise_constant(global_step1, boundaries2, values2)

    opt_stable = tf.train.AdamOptimizer(learning_rate)
    opt_finetuning = tf.train.AdamOptimizer(learning_rate1)
    images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           Config.channels,
                                                           Config.image_size,
                                                           Config.image_size,
                                                           3))
    labels_placeholder = tf.placeholder(tf.int64, shape=(None))
    keep_prob = tf.placeholder(tf.float32)

    train_data = dataset_records.dataset_records(TRAIN_PATH, BATCH_SIZE, EPOCH)
    c3d = C3d(num_class=NUM_CLASS, keep_prob=keep_prob)
    logit = c3d.build_model(images_placeholder)
    weights = c3d.getweights()
    biases = c3d.getbiases()
    accuracy = tower_acc(logit, labels_placeholder)

    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logit)
    )

    weight_decay_loss = tf.get_collection('weight_decay_loss')
    weight_decay_loss = tf.add_n(weight_decay_loss)
    loss = tf.add(cross_entropy_mean, weight_decay_loss)
    varlist2 = [weights['out'], biases['out']]
    varlist1 = list(set(list(weights.values()) + list(biases.values())) - set(varlist2))
    grads1 = opt_stable.compute_gradients(loss, varlist1)
    grads2 = opt_finetuning.compute_gradients(loss, varlist2)
    apply_gradient_op1 = opt_stable.apply_gradients(grads1, global_step=global_step)
    apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step1)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step=global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(list(weights.values()) + list(biases.values()))
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    sess.run(init)
    if os.path.isfile(FINE_PATH + ".data-00000-of-00001"):
        saver.restore(sess, FINE_PATH)
    # merged = tf.summary.merge_all()

    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    cross_loss_summary = tf.summary.scalar(
        'cross_loss',
        cross_entropy_mean
    )
    weight_decay_loss_summary = tf.summary.scalar('weight_decay_loss', weight_decay_loss)
    total_loss_summary = tf.summary.scalar('total_loss', loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(TENSORBOARD_PATH + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(TENSORBOARD_PATH + 'test', sess.graph)
    train_iterator = train_data.make_one_shot_iterator()
    train_one_element = train_iterator.get_next()

    step = 0
    best_acc = 0
    best_model_index = -1
    best_model_index_file = open("best_model_index.tx", "wb+")
    test_acc_file = open("test_acc.tx", "wb+")
    try:
        while True:
            data = sess.run(train_one_element)

            if (step) % STEP_INV1 == 0:
                print('Training Data Eval:')
                summary, acc, cross_loss_val, _ = sess.run(
                    [merged, accuracy, cross_entropy_mean, train_op],
                    feed_dict={images_placeholder: data[0],
                               labels_placeholder: data[1],
                               keep_prob: 0.6,
                               })
                print ("{0}:train accuracy: {1:.5f}   loss: {2:.5f}".format(step, acc, cross_loss_val))
                train_writer.add_summary(summary, step)
            else:
                sess.run(train_op, feed_dict={
                    images_placeholder: data[0],
                    labels_placeholder: data[1],
                    keep_prob: 0.6,
                })
            if step % STEP_INV2 == 0 and step != 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, 'c3d_ucf_model'), global_step=step)
                test_data = dataset_records.dataset_records(TEST_PATH, BATCH_SIZE, 1)
                test_iterator = test_data.make_one_shot_iterator()
                test_one_element = test_iterator.get_next()
                print('Validation Data Eval:')
                accuracy_total = 0
                c_loss_total = 0
                len_total = 0
                test_step = 0
                try:
                    while True:
                        test_data = sess.run(test_one_element)
                        acc, c_loss = sess.run(
                            [accuracy, cross_entropy_mean],
                            feed_dict={
                                images_placeholder: test_data[0],
                                labels_placeholder: test_data[1],
                                keep_prob: 1,
                            })
                        len_total += len(test_data[1])
                        c_loss_total += c_loss
                        accuracy_total += acc * len(test_data[1])
                        test_step += 1

                except tf.errors.OutOfRangeError:
                    acc_avg = accuracy_total / len_total
                    c_loss_avg = c_loss_total / test_step
                    print (
                        "{0}:test accuracy:{1:.5f} loss:{2:.5f}".format(step, acc_avg, c_loss_avg))
                    test_acc_file.write("{0}:test accuracy:{1:.5f} loss:{2:.5f}\n".format(step, acc_avg, c_loss_avg))
                    test_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=acc_avg)]),
                        step)
                    test_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='cross_loss', simple_value=c_loss_avg)]),
                        step)
                    if acc_avg > best_acc:
                        saver.save(sess, os.path.join(MODEL_BEST_PATH, 'c3d_ucf_model_best'))
                        best_acc = acc_avg
                        best_model_index = step
                        print("best model index is:{},acc:{}".format(best_model_index, acc_avg))
                        best_model_index_file.write("best model index is:{},acc:{}\n".format(best_model_index, acc_avg))

            step += 1
    except tf.errors.OutOfRangeError:
        best_model_index_file.flush()
        best_model_index_file.close()
        test_acc_file.flush()
        test_acc_file.close()
        print("best model index is:{},acc:{}".format(best_model_index, best_acc))
        print("end!")
