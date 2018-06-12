#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 2018

@author: hehuihui1994 (he__huihui@163.com)
https://github.com/hehuihui1994
"""

from PrepareData import load_data
from Evalue import Emo_eval_cause
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class LR(object):
    
    def __init__(self, input_file, test_index, n_class, embedding_dim):
        self.input_file = input_file,
        self.test_index = test_index,
        self.n_class = n_class,
        self.embedding_dim = embedding_dim

    def scale(self):
        sc = StandardScaler()
        sc.fit(self.x1)
        sc.mean_
        sc.scale_

        # 规范化训练集
        self.x1 = sc.transform(self.x1)
        # 同样的标准，规范化测试集
        self.x2 = sc.transform(self.x2)

    

    def run(self):
        print "step 1 : load data ..."
        self.x1, self.y1, self.x2, self.y2 = load_data(self.input_file[0], self.test_index[0], self.n_class, self.embedding_dim)

        # print self.x1[0]
        # print len(self.x2)
        # print len(self.y2)
        
        print "step 2 : scale data ..."
        self.scale()

        # # print self.x1[0]

        print "step 3 : logistic regression ..."
        lr = LogisticRegression()
        lr.fit(self.x1, self.y1)

        print "step 4 : predict ... "
        # print len(self.x2)
        y_pred_cause = lr.predict(self.x2)
        # print len(y_pred_cause)
        y_true_cause = self.y2
        # print len(self.y2)
        Emo_eval_cause(y_true_cause, y_pred_cause)


if __name__ == '__main__':
    for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    # for j in [1]:
        print('hhhe_fold{}.......'.format(j))
        obj = LR(
            input_file = '../resources/Emo_Cause_corpus_HHH_simple_dis_lr.txt',
            test_index = j,
            n_class = 2,
            embedding_dim = 200,
        )
        obj.run()
        



