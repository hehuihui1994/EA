#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 2017

@author: hehuihui1994 (he__huihui@163.com)
https://github.com/hehuihui1994
"""

import sys
import time
import os
import numpy as np
import tensorflow as tf
from Prepare import batch_index, load_data
from Evalu import Evaluate

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 20, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
# tf.app.flags.DEFINE_float('keep_prob', 1.0, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.0005, 'l2 regularization')

tf.app.flags.DEFINE_integer('display_step', 1, 'number of test display step')
tf.app.flags.DEFINE_integer('training_iter', 100, 'number of train iter')
# tf.app.flags.DEFINE_integer('embedding_dim', 100, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('hidden', 60, 'number of hidden')
tf.app.flags.DEFINE_integer('n_class', 6, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_doc_len', 1134, 'max number of tokens per sentence')

tf.app.flags.DEFINE_string('train_file_path', 'train.sample', 'training file')
tf.app.flags.DEFINE_string('test_file_path', 'test.sample', 'testing file')
# tf.app.flags.DEFINE_string('embedding_file_path', 'zh_embedding_100.txt', 'embedding file')
tf.app.flags.DEFINE_string('test_index', 1, 'test_index')
# tf.app.flags.DEFINE_string('embedding_type', 0, 'embedding_type')

class NN(object):

    def __init__(self,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                #  keep_prob=FLAGS.keep_prob,
                 l2_reg=FLAGS.l2_reg,
                 display_step=FLAGS.display_step,
                 training_iter=FLAGS.training_iter,
                #  embedding_dim=FLAGS.embedding_dim,
                 hidden=FLAGS.hidden,
                 n_class=FLAGS.n_class,
                 max_doc_len=FLAGS.max_doc_len,
                 train_file_path=FLAGS.train_file_path,
                 test_file_path=FLAGS.test_file_path,    
                #  w2v_file=FLAGS.embedding_file_path,
                 test_index=FLAGS.test_index,
                #  embedding_type=FLAGS.embedding_type,
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # self.Keep_Prob = keep_prob
        self.l2_reg = l2_reg

        self.display_step = display_step
        self.training_iter = training_iter
        # self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.n_class = n_class
        self.max_doc_len = max_doc_len

        self.train_file_path = train_file_path
        self.test_file_path = test_file_path        
        # self.w2v_file = w2v_file
        self.test_index = test_index
        # self.embedding_type = embedding_type


        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.max_doc_len])
            self.y = tf.placeholder(tf.float32, [None, self.n_class])

        def init_variable(shape):
            initial = tf.random_uniform(shape, -0.01, 0.01)
            return tf.Variable(initial)

        with tf.name_scope('weights'):
            self.weights = {
                'w': init_variable([self.max_doc_len, self.hidden]),
                'softmax': init_variable([self.hidden, self.n_class]),
            }

        with tf.name_scope('biases'):
            self.biases = {
                'b': init_variable([self.hidden]),
                'softmax': init_variable([self.n_class]),
            }

    def model(self, inputs):
        # inputs = tf.reshape(inputs, [-1, self.max_doc_len])
        outputs = tf.sigmoid(tf.matmul(inputs, self.weights['w'])+self.biases['b'])
        predict = tf.matmul(outputs, self.weights['softmax']) + self.biases['softmax']
        predict = tf.nn.softmax(predict)

        return predict


    def run(self):
        # inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        inputs = self.x
        prob = self.model(inputs)

        with tf.name_scope('loss'):
            cost = - tf.reduce_mean(self.y * tf.log(prob))

            reg = tf.nn.l2_loss(self.weights['w'])  \
            +tf.nn.l2_loss(self.biases['b'])\
            +tf.nn.l2_loss(self.weights['softmax'])\
            +tf.nn.l2_loss(self.biases['softmax'])
            cost += reg*self.l2_reg



        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        with tf.name_scope('summary'):
            localtime = time.strftime("%X %Y-%m-%d", time.localtime())
            Summary_dir = 'Summary/' + localtime

            info = 'batch-{}, lr-{}, l2_reg-{}'.format(
                self.batch_size,  self.learning_rate, self.l2_reg)
        
            summary_acc = tf.summary.scalar('ACC ' , accuracy)
            summary_loss = tf.summary.scalar('LOSS ' , cost)
            summary_op = tf.summary.merge([summary_loss, summary_acc])

            test_acc = tf.placeholder(tf.float32)
            test_loss = tf.placeholder(tf.float32)
            summary_test_acc = tf.summary.scalar('ACC ' , test_acc)
            summary_test_loss = tf.summary.scalar('LOSS ', test_loss)
            summary_test = tf.summary.merge(
                [summary_test_loss, summary_test_acc])

            train_summary_writer = tf.summary.FileWriter(
                Summary_dir + '/train')
            test_summary_writer = tf.summary.FileWriter(Summary_dir + '/test')

        with tf.name_scope('saveModel'):
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            save_dir = 'Models/' + localtime + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # with tf.name_scope('readData'):
        print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
        tr_x, tr_y = load_data(
            self.train_file_path,
            self.max_doc_len,
            self.n_class
        )
        te_x, te_y = load_data(
            self.test_file_path,
            self.max_doc_len,
            self.n_class,
            test = True
        )
        print tr_x.shape
        print tr_y.shape
        print 'train docs: {}    test docs: {}'.format(len(tr_y), len(te_y))
        print 'training_iter:', self.training_iter
        print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            max_acc, bestIter = 0., 0

            def test():
                acc, loss, cnt = 0., 0., 0
                for test, num in self.get_batch_data(te_x, te_y, self.batch_size, test = True):
                    _loss, _acc = sess.run([cost, correct_num], feed_dict=test)
                    acc += _acc
                    loss += _loss * num
                    cnt += num
                loss = loss / cnt
                acc = acc / cnt
                return loss, acc

            def new_test():
                feed_dict = {
                    self.x: te_x,
                }
                y_true = te_y
                y_pred_p = sess.run(prob, feed_dict=feed_dict)
                Evaluate(y_pred_p, y_true)

            for i in xrange(self.training_iter):

                for train, _ in self.get_batch_data(tr_x, tr_y, self.batch_size):
                    _, step, summary, loss, acc = sess.run(
                        [optimizer, global_step, summary_op, cost, accuracy], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)

                if i % self.display_step == 0:
                    loss, acc=test()

                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                        saver.save(sess, save_dir, global_step=step)
                        new_test()

                    summary = sess.run(summary_test, feed_dict={
                                       test_loss: loss, test_acc: acc})
                    test_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: test loss={:.6f}, test acc={:.6f}'.format(step, loss, acc)
                    print 'round {}: max_acc={} BestIter={}\n'.format(i, max_acc, bestIter)

            print 'Optimization Finished!'


    def get_batch_data(self, x, y, batch_size, test=False):
        for index in batch_index(len(y), batch_size, 1, test):
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
            }
            yield feed_dict, len(index)



def main(_):
    obj = NN()
    obj.run()
    
    
if __name__ == '__main__':
    tf.app.run()
