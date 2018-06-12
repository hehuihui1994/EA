#!/usr/bin/env python
# encoding: utf-8


import sys
import time
import datetime
import os
import numpy as np
import tensorflow as tf
from PrepareData import batch_index, load_w2v, load_data, prob2one_hot
from Evalue import Emo_eval, Emo_eval_emotion

#预定义超参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.05, 'learning rate')
# tf.app.flags.DEFINE_float('keep_prob', 1.0, 'softmax layer dropout keep prob')

tf.app.flags.DEFINE_float('keep_prob1', 1, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1, 'softmax layer dropout keep prob')

tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')

#隐藏层单元个数
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')


tf.app.flags.DEFINE_integer('display_step', 1, 'number of test display step')
tf.app.flags.DEFINE_integer('training_iter', 20, 'number of train iter')

#  class label
tf.app.flags.DEFINE_integer('n_doc_class', 6, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_sentence_class', 2, 'number of distinct class')


tf.app.flags.DEFINE_integer('max_sentence_len', 50, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 30, 'max number of tokens per documents')

# file path
tf.app.flags.DEFINE_string('train_file_path', '../resources/Emo_Cause_corpus_HHH_simple_whole.txt', 'training file')
tf.app.flags.DEFINE_string('embedding_file_path', '../resources/embeddings/vectors_iter15.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')

tf.app.flags.DEFINE_string('test_index', 1, 'test_index')
tf.app.flags.DEFINE_string('embedding_type', 0, 'embedding_type')

class HN_BiLSTM_ATT(object):

    def __init__(self,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,

                 keep_prob1 = FLAGS.keep_prob1,
                 keep_prob2 = FLAGS.keep_prob2,

                 l2_reg=FLAGS.l2_reg,
                 
                 n_hidden = FLAGS.n_hidden,

                 display_step=FLAGS.display_step,
                 training_iter=FLAGS.training_iter,

                 n_doc_class = FLAGS.n_doc_class,
                 n_sentence_class = FLAGS.n_sentence_class,

                 max_sentence_len = FLAGS.max_sentence_len,
                 max_doc_len = FLAGS.max_doc_len,                 
                 
                 train_file_path=FLAGS.train_file_path,
                 w2v_file=FLAGS.embedding_file_path,
                 embedding_dim=FLAGS.embedding_dim,
                 
                 test_index=FLAGS.test_index,
                 embedding_type=FLAGS.embedding_type,
                 scope='test'                 
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.keep_prob1 = keep_prob1
        self.keep_prob2 = keep_prob2

        self.l2_reg = l2_reg

        self.n_hidden = n_hidden

        self.display_step = display_step
        self.training_iter = training_iter

        self.n_doc_class = n_doc_class
        self.n_sentence_class = n_sentence_class

        self.max_sentence_len = max_sentence_len
        self.max_doc_len = max_doc_len

        self.train_file_path = train_file_path
        self.w2v_file = w2v_file
        self.embedding_dim = embedding_dim

        self.test_index = test_index
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

        #定义需feed数据的tensor
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_doc_len, self.max_sentence_len])
            # self.y = tf.placeholder(tf.float32, [None, self.n_class])

            self.y_doc = tf.placeholder(tf.float32, [None, self.n_doc_class])
            # self.y_sentence = tf.placeholder(tf.float32, [None, self.n_sentence_class])
            
            self.sen_len = tf.placeholder(tf.int32, [None, self.max_doc_len])
            self.doc_len = tf.placeholder(tf.int32, None)

            self.keep_prob1 = tf.placeholder(tf.float32)
            self.keep_prob2 = tf.placeholder(tf.float32)

        def init_variable(shape):
            initial = tf.random_uniform(shape, -0.01, 0.01)
            return tf.Variable(initial)

       #所有的模型可训练参数
        with tf.name_scope('weights'):
            self.weights = {
                # Attention sentence
                'w_1': tf.Variable(tf.random_uniform([2 * self.n_hidden, 2 * self.n_hidden], -0.01, 0.01)),
                'u_1': tf.Variable(tf.random_uniform([2 * self.n_hidden, 1], -0.01, 0.01)),
                # Attention doc
                'w_2': tf.Variable(tf.random_uniform([2 * self.n_hidden, 2 * self.n_hidden], -0.01, 0.01)),
                'u_2': tf.Variable(tf.random_uniform([2 * self.n_hidden, 1], -0.01, 0.01)),

                # softmax sentence
                # 'softmax_sentence': init_variable([self.n_hidden*2, self.n_sentence_class]),
                # softmax doc
                'softmax_doc': init_variable([self.n_hidden*2, self.n_doc_class]),

            }

        with tf.name_scope('biases'):
            self.biases = {
                #Attention
                'w_1': tf.Variable(tf.random_uniform([2 * self.n_hidden], -0.01, 0.01)),
                'w_2': tf.Variable(tf.random_uniform([2 * self.n_hidden], -0.01, 0.01)),                

                # 'softmax_sentence': init_variable([self.n_sentence_class]),
                'softmax_doc': init_variable([self.n_doc_class]),
            }
    
    def biLSTM(self,inputs,sequence_length,n_hidden,scope):
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
            cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            scope=scope
        )
        return tf.concat(outputs, 2)

    #定义模型计算流程，即从输入计算预测的过程
    def model(self, inputs):
        # inputs, [-1, self.max_sentence_len, self.embedding_dim]
        # inputs 已经被reshape了
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)

        # [[1,2,3],[4,5,6]]  ->  [1,2,3,4,5,6,] 
        # [doc1, doc2] 
        # tmp_sen_len = [0] * max_doc_len
        # reshape之后变成正常的单层模型
        sen_len = tf.reshape(self.sen_len, [-1])
        
        #词语级编码
        with tf.name_scope('word_encode'):
            outputs = self.biLSTM(inputs, sen_len, n_hidden=self.n_hidden, scope=self.scope+'word_layer_last')


        batch_size = tf.shape(outputs)[0]
        #使用注意力机制得到句子表示
        with tf.name_scope('word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.n_hidden])
            u = tf.tanh(tf.matmul(output, self.weights['w_1']) + self.biases['w_1'])
            alpha = tf.reshape(tf.matmul(u, self.weights['u_1']), [batch_size, 1, self.max_sentence_len])  # batch_size * 1 * n_step
            alpha = self.softmax(alpha, self.sen_len, self.max_sentence_len)
            outputs = tf.matmul(alpha, outputs)  # batch_size * 1 * 2n_hidden
        
        # 按照max_doc_len还原
        # [1,2,3,4,5,6]   ->   [[1,2,3],[4,5,6]]
        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2*self.n_hidden])


        #句子编码
        with tf.name_scope('sentence_encode'):
            outputs = self.biLSTM(outputs, self.doc_len, n_hidden=self.n_hidden, scope=self.scope+'sentence')

        batch_size = tf.shape(outputs)[0]
        #使用注意力机制得到文档表示
        with tf.name_scope('sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.n_hidden])
            u = tf.tanh(tf.matmul(output, self.weights['w_2']) + self.biases['w_2'])
            alpha = tf.reshape(tf.matmul(u, self.weights['u_2']), [batch_size, 1, self.max_doc_len])  # batch_size * 1 * n_step
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)  # batch_size * 1 * 2n_hidden

        #softmax层
        with tf.name_scope('softmax_doc'):
            outputs = tf.reshape(outputs, [batch_size, 2 * self.n_hidden])
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob2)
            predict_doc = tf.matmul(outputs, self.weights['softmax_doc']) + self.biases['softmax_doc']
            predict_doc = tf.nn.softmax(predict_doc)

        return predict_doc
    
    #自定义softmax函数，用于attention计算
    def softmax(self, inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.exp(inputs)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    # 训练模型
    def run(self):
        #   x_i   tmp_x = np.zeros((max_doc_len, max_sen_len), dtype=np.int)
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        # 去掉max_doc_len 这一层次结构
        inputs = tf.reshape(inputs, [-1, self.max_sentence_len, self.embedding_dim])        
        prob_doc = self.model(inputs)

        #定义损失和l2正则化项
        with tf.name_scope('loss'):
            cost = - tf.reduce_mean(self.y_doc * tf.log(prob_doc))

            reg, variables = tf.nn.l2_loss(self.word_embedding), []

            # variables.append('softmax_sentence')
            variables.append('softmax_doc')
                
            for vari in variables:
                reg += tf.nn.l2_loss(self.weights[vari]) + \
                    tf.nn.l2_loss(self.biases[vari])

            # add attention parameters
            reg += tf.nn.l2_loss(self.weights['w_1']) + tf.nn.l2_loss(self.biases['w_1'])
            reg += tf.nn.l2_loss(self.weights['u_1'])   

            reg += tf.nn.l2_loss(self.weights['w_2']) + tf.nn.l2_loss(self.biases['w_2'])
            reg += tf.nn.l2_loss(self.weights['u_2'])               

            cost += reg * self.l2_reg

        #定义optimizer，即优化cost的节点
        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

            
        #计算模型预测准确率
        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob_doc, 1), tf.argmax(self.y_doc, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        #启用sunmary，可视化训练过程
        with tf.name_scope('summary'):
            localtime = time.strftime("%X %Y-%m-%d", time.localtime())
            Summary_dir = 'Summary/' + localtime

            info = 'batch-{}, lr-{}, kb-{}, l2_reg-{}'.format(
                self.batch_size,  self.learning_rate, self.keep_prob1, self.l2_reg)
            info = info + '\ntrain_file_path:' + self.train_file_path + '\ntest_index:' + str(self.test_index) + '\nembedding_type:' + str(self.embedding_type) + '\nMethod: Emotion_CNN_prob'
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
            tr_x, tr_y, tr_sen_len, tr_doc_len, te_x, te_y, te_sen_len, te_doc_len= load_data(
                self.train_file_path,
                self.word_id_mapping,
                self.max_sentence_len,
                self.max_doc_len,
                self.test_index,
                self.n_doc_class
            )
            print 'train docs: {}    test docs: {}'.format(len(tr_y), len(te_y))
            print 'training_iter:', self.training_iter
            print info
            print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.initialize_all_variables())
            max_acc, bestIter = 0., 0


            def test():
                feed_dict = {
                    self.x: te_x,
                    self.y_doc: te_y,
                    self.sen_len: te_sen_len,
                    self.doc_len: te_doc_len,
                    # self.keep_prob: 1.0,
                    self.keep_prob1: FLAGS.keep_prob1,
                    self.keep_prob2: FLAGS.keep_prob2,
                }
                loss, acc = sess.run([cost, accuracy], feed_dict=feed_dict)
                return loss, acc
                

            def new_test():
                feed_dict = {
                    self.x: te_x,
                    self.y_doc: te_y,
                    self.sen_len: te_sen_len,
                    self.doc_len: te_doc_len,
                    # self.keep_prob: 1.0,
                    self.keep_prob1: FLAGS.keep_prob1,
                    self.keep_prob2: FLAGS.keep_prob2,
                }

                y_pred_doc_p = sess.run(prob_doc, feed_dict=feed_dict)

                y_pred_doc = prob2one_hot(y_pred_doc_p, self.n_doc_class)

                y_true_doc = te_y


                ''' test '''

                print('y_true_doc[0] : {}'.format(y_true_doc[0]))
                print('y_pred_doc_p[0] : {}'.format(y_pred_doc_p[0]))

                print('y_pred_doc[0] : {}'.format(y_pred_doc[0]))                

                Emo_eval_emotion(y_true_doc, y_pred_doc)




            for i in xrange(self.training_iter):
                
                # starttime = datetime.datetime.now()

                for train, _ in self.get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, self.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2, test=False):
                    # print train
                    _, step, summary, loss, acc = sess.run(
                        [optimizer, global_step, summary_op, cost, accuracy], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)
                
                # endtime = datetime.datetime.now()
                # runtime = (endtime-starttime).seconds
                # print "time cost = {}".format(runtime)

                if i % self.display_step == 0:
                    
                    loss, acc= test()

                    if acc > max_acc:
                        new_test()
                        max_acc = acc
                        bestIter = step
                        saver.save(sess, save_dir, global_step=step)                

                    summary = sess.run(summary_test, feed_dict={
                                       test_loss: loss, test_acc: acc})
                    test_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: test loss={:.6f}, test acc={:.6f}'.format(step, loss, acc)                
                    print 'round {}: max_acc={} BestIter={}\n'.format(i, max_acc, bestIter)
            
            # new_test()

            print 'Optimization Finished!'

    #获取batch数据
    def get_batch_data(self, x, y, sen_len, doc_len, batch_size, keep_prob1, keep_prob2, test= False):
        for index in batch_index(len(y), batch_size, 1, test):
            feed_dict = {
                self.x: x[index],
                self.y_doc: y[index],
                self.sen_len: sen_len[index],
                self.doc_len: doc_len[index],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,   
            }
            yield feed_dict, len(index)



def main(_):
    for i in [0]:
        # for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for j in [10]:
            print('hhhe_ebdt{}_index{}.txt'.format(i, j))
            #sys.stdout = open('ebdt'+str(i)+'_index'+str(j)+'.txt', 'w')
            g = tf.Graph()
            with g.as_default():
                obj = HN_BiLSTM_ATT(
                    test_index=j,
                    embedding_type=i,
                    scope='hhhe_ebdt{}_index{}.txt'.format(i, j)
                )
                obj.run()
    
if __name__ == '__main__':
    tf.app.run()
