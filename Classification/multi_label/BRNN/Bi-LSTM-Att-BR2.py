#!/usr/bin/env python
# encoding: utf-8


'''
两种实现方式：
    1）training epoch 更新所有的二分类器  
    2）training epoch * training_iter 训练好一个分类器后，再训练另外一个分类器(*)

    terminate

需要计算得到 Time cost 以及 Hamming Loss  

best iter  =  2

HL :  0.13241616509
RL :  0.155782117949
OE :  0.412439094296
Coverage :  1.82143880768
AP :  0.709514255688 
'''


import sys
import time
import datetime
import os
import numpy as np
import tensorflow as tf
from PrepareData import batch_index, load_w2v, load_data
from Evalue import Emotion_eval, Emo_eval
# from Evalue import calibrated_label_ranking
from sklearn.metrics import label_ranking_average_precision_score

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 256, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.0001, 'l2 regularization')

#隐藏层单元个数
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')

tf.app.flags.DEFINE_integer('display_step', 1, 'number of test display step')
tf.app.flags.DEFINE_integer('training_iter', 2, 'number of train iter')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_doc_len', 90, 'max number of tokens per sentence')

tf.app.flags.DEFINE_string('train_file_path', '../resources/renCECps_multi_class_probability_9.txt', 'training file')
tf.app.flags.DEFINE_string('embedding_file_path', '../resources/vectors_iter15.txt', 'embedding file')
tf.app.flags.DEFINE_string('test_index', 1, 'test_index')
tf.app.flags.DEFINE_string('label_index', 1, 'label_index')
tf.app.flags.DEFINE_string('embedding_type', 0, 'embedding_type')

class Bi_LSTM_Att_BC_model(object):

    def __init__(self,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 keep_prob=FLAGS.keep_prob,
                 l2_reg=FLAGS.l2_reg,

                 #
                 n_hidden = FLAGS.n_hidden,

                 display_step=FLAGS.display_step,
                 training_iter=FLAGS.training_iter,
                 embedding_dim=FLAGS.embedding_dim,
                 n_class=FLAGS.n_class,
                 max_doc_len=FLAGS.max_doc_len,
                 train_file_path=FLAGS.train_file_path,
                 w2v_file=FLAGS.embedding_file_path,
                 test_index=FLAGS.test_index,
                 label_index = FLAGS.label_index,
                 embedding_type=FLAGS.embedding_type,
                 scope='test'
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.Keep_Prob = keep_prob
        self.l2_reg = l2_reg

         #
        self.n_hidden = n_hidden

        self.display_step = display_step
        self.training_iter = training_iter
        self.embedding_dim = embedding_dim
        self.n_class = n_class
        self.max_doc_len = max_doc_len

        self.train_file_path = train_file_path
        self.w2v_file = w2v_file
        self.test_index = test_index
        self.label_index = label_index
        self.embedding_type = embedding_type
        self.scope=scope

        self.word_id_mapping, self.w2v = load_w2v(
            self.w2v_file, self.embedding_dim)
        if embedding_type == 0:  # Pretrained and Untrainable
            self.word_embedding = tf.constant(
                self.w2v, dtype=tf.float32, name='word_embedding')
        elif embedding_type == 1:  # Pretrained and Trainable
            self.word_embedding = tf.Variable(
                self.w2v, dtype=tf.float32, name='word_embedding')
        elif embedding_type == 2:  # Random and Trainable
            self.word_embedding = tf.Variable(tf.random_uniform(
                [len(self.word_id_mapping) + 1, self.embedding_dim], -0.1, 0.1), name='word_embedding')

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_doc_len])
            self.y = tf.placeholder(tf.float32, [None, self.n_class])
            self.doc_len = tf.placeholder(tf.int32, None)
            self.keep_prob = tf.placeholder(tf.float32)

        def init_variable(shape):
            initial = tf.random_uniform(shape, -0.01, 0.01)
            return tf.Variable(initial)

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': init_variable([self.n_hidden*2, self.n_class]),
                 #Attention
                'w_1': tf.Variable(tf.random_uniform([self.n_hidden*2, self.n_hidden*2], -0.01, 0.01)),
                #alpha
                'u_1': tf.Variable(tf.random_uniform([self.n_hidden*2, 1], -0.01, 0.01)),
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': init_variable([self.n_class]),
                 #
                'w_1': tf.Variable(tf.random_uniform([self.n_hidden*2], -0.01, 0.01)),
            }

    def model(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.max_doc_len, self.embedding_dim])
        #Bi - LSTM
        with tf.name_scope('doc_encode'):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.contrib.rnn.LSTMCell(self.n_hidden),
                cell_bw=tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=inputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope=self.scope)
            outputs = tf.concat(outputs, 2)
            
            # outputs = tf.concat(state[1],1)

        batch_size = tf.shape(outputs)[0]

        with tf.name_scope('sentence_attention'):
            # Attention
            output = tf.reshape(outputs, [-1, self.n_hidden*2])
            u = tf.tanh(tf.matmul(output, self.weights['w_1']) + self.biases['w_1'])
            alpha = tf.reshape(tf.matmul(u, self.weights['u_1']), [batch_size, 1, self.max_doc_len])  # batch_size * 1 * n_step                    
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)

            outputs = tf.reshape(tf.matmul(alpha, outputs), [-1,self.n_hidden*2])  # batch_size * 1 * n_hidden   V batch_size  * n_hidden

        with tf.name_scope('softmax'):
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
            predict = tf.matmul(outputs, self.weights['softmax']) + self.biases['softmax']
            predict = tf.nn.softmax(predict)

        return predict

    def softmax(self, inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.exp(inputs)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    def run(self):
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        prob = self.model(inputs)

        with tf.name_scope('loss'):
            cost = - tf.reduce_mean(self.y * tf.log(prob))
            reg, variables = tf.nn.l2_loss(self.word_embedding), ['softmax']
            for vari in variables:
                reg += tf.nn.l2_loss(self.weights[vari]) + \
                    tf.nn.l2_loss(self.biases[vari])
            # add attention parameters
            reg += tf.nn.l2_loss(self.weights['w_1']) + tf.nn.l2_loss(self.biases['w_1'])
            reg += tf.nn.l2_loss(self.weights['u_1']) 
            cost += reg * self.l2_reg

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

            info = 'batch-{}, lr-{}, kb-{}, l2_reg-{}'.format(
                self.batch_size,  self.learning_rate, self.Keep_Prob, self.l2_reg)
            info = info + '\ntrain_file_path:' + self.train_file_path + '\ntest_index:' + str(self.test_index) + '\nembedding_type:' + str(self.embedding_type) + '\nMethod: Emotion_GRU'
            summary_acc = tf.summary.scalar('ACC ' + info, accuracy)
            summary_loss = tf.summary.scalar('LOSS ' + info, cost)
            summary_op = tf.summary.merge([summary_loss, summary_acc])

            test_acc = tf.placeholder(tf.float32)
            test_loss = tf.placeholder(tf.float32)
            summary_test_acc = tf.summary.scalar('ACC ' + info, test_acc)
            summary_test_loss = tf.summary.scalar('LOSS ' + info, test_loss)
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

        with tf.name_scope('readData'):
            print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
            tr_x, tr_y, tr_doc_len, te_x, te_y, te_doc_len = load_data(
                self.train_file_path,
                self.word_id_mapping,
                self.max_doc_len,
                self.test_index,
                self.label_index,
                self.n_class,
            )
            print 'train docs: {}    test docs: {}'.format(len(tr_y), len(te_y))
            print 'training_iter:', self.training_iter
            print info
            print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        # with tf.Session() as sess:
        with tf.Session(config=tf_config) as sess:        
            sess.run(tf.initialize_all_variables())
            max_acc, bestIter = 0., 0
            max_AP = 0.            

            def test():
                feed_dict = {
                    self.x: te_x,
                    self.doc_len: te_doc_len,
                    self.keep_prob: 1.0,
                }
                # y_true = te_y
                # y_pred_p = sess.run(prob, feed_dict=feed_dict)
                # AP = label_ranking_average_precision_score(y_true, y_pred_p) 
                # return AP  
                acc, loss, cnt = 0., 0., 0
                for test, num in self.get_batch_data(te_x, te_y, te_doc_len, 20, keep_prob=1.0, test=True):
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
                    self.doc_len: te_doc_len,
                    self.keep_prob: 1.0,
                }
                y_true = te_y
                y_pred_p = sess.run(prob, feed_dict=feed_dict)
                return y_true, y_pred_p

            if self.training_iter==0:
                # saver.restore(sess, 'Models/10:01:44 2017-03-11/-856')
                loss, acc=test()
                # AP = test()
                print "loss : {}, acc : {}".format(loss,acc)
                # print "AP : ",AP
                # new_test()

            # For_calibrated_B = np.loadtxt('For_calibrated_B'+str(self.test_index)+'.txt', delimiter=',')  
      
            
            cur_y_true, best_y_pred_p = [], []
            runtime = 0.
            for i in xrange(self.training_iter):
                
                starttime = datetime.datetime.now()

                for train, _ in self.get_batch_data(tr_x, tr_y, tr_doc_len, self.batch_size, self.Keep_Prob,test=False):
                    _, step, summary, loss, acc = sess.run(
                        [optimizer, global_step, summary_op, cost, accuracy], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)

                endtime = datetime.datetime.now()
                runtime = (endtime-starttime).seconds
                print "time cost = {}".format(runtime)

                if i % self.display_step == 0:
                    loss, acc=test()
                    # AP = test()

                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                        # saver.save(sess, save_dir, global_step=step)
                        # cur_y_true, best_y_pred_p = new_test()
                    # if AP > max_AP:
                    #     max_AP = AP
                    #     bestIter = step
                    #     saver.save(sess, save_dir, global_step=step)
                    #     new_test()                    

                    summary = sess.run(summary_test, feed_dict={
                                       test_loss: loss, test_acc: acc})
                    test_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: test loss={:.6f}, test acc={:.6f}'.format(step, loss, acc)
                    print 'round {}: max_acc={} BestIter={}\n'.format(i, max_acc, bestIter)
                    # print 'Iter {}: test AP={:.6f}'.format(step, AP)                    
                    # print 'round {}: max_AP={} BestIter={}\n'.format(i, max_AP, bestIter)

            cur_y_true, best_y_pred_p = new_test()

            print 'Optimization Finished!'
            return runtime, cur_y_true, best_y_pred_p
        

    def get_batch_data(self, x, y, doc_len, batch_size, keep_prob, test= False):
        for index in batch_index(len(y), batch_size, 1, test):
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.doc_len: doc_len[index],
                self.keep_prob: keep_prob,
            }
            yield feed_dict, len(index)



def main(_):
    for i in [0]:
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # for j in [1, 2, 3, 4, 5]: 
        # for j in [6, 7, 8, 9, 10]:  
            print('hhhe_ebdt{}_index{}.txt'.format(i, j))
            #sys.stdout = open('ebdt'+str(i)+'_index'+str(j)+'.txt', 'w')
            # for each label
            y_true = []
            y_pred_p = []
            y_pred = []
            sum_time_cost = 0.

            for k in range(9):
                print('hhhe_ebdt{}_index{}_label{}.txt'.format(i, j, k))

                g = tf.Graph()
                with g.as_default():
                
                    obj = Bi_LSTM_Att_BC_model(
                        test_index=j,
                        embedding_type=i,
                        label_index = k,
                        scope='hhhe_ebdt{}_index{}_label{}.txt'.format(i, j, k)
                    )
                    time_cost, cur_y_true, best_y_pred_p  = obj.run()

                sum_time_cost += time_cost

                # print "cur_y_true[0] : "
                print cur_y_true[0]
                # print "best_y_pred_p[9] : "
                print best_y_pred_p[0]                

                cur_y_true = np.argmax(cur_y_true, axis = 1).reshape(-1,1)
                best_y_pred = np.argmax(best_y_pred_p, axis = 1).reshape(-1,1)

                best_y_pred_p = best_y_pred_p[:,1].reshape(-1,1)


                if k==0:
                    y_true = cur_y_true
                    y_pred = best_y_pred
                    y_pred_p = best_y_pred_p
                else:
                    y_true = np.concatenate((y_true, cur_y_true), axis=1)
                    y_pred = np.concatenate((y_pred, best_y_pred), axis=1)
                    y_pred_p = np.concatenate((y_pred_p, best_y_pred_p), axis=1)


            Emotion_eval(y_true, y_pred, y_pred_p)
            Emo_eval(y_true, y_pred)
            print y_true[0]
            print y_pred[0]
            print y_pred_p[0]
            
            # np.savetxt('For_calibrated_B'+str(j)+'_share2.txt', y_pred, fmt="%d", delimiter=",")

            print "Finish the {}th experiment!".format(j)
            print "sum of the time cost for each binary classifier: {}".format(sum_time_cost)
    
if __name__ == '__main__':
    tf.app.run()
