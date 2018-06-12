#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: hehuihui1994 (he__huihui@163.com)
https://github.com/hehuihui1994
"""

import re
import sys
import random
import numpy as np


def load_data(input_file, max_doc_len=4000 + 1):

    x= []

    print 'loading input {}...'.format(input_file)
    read_index = 0
    for line in open(input_file).readlines():
        if read_index%100 == 0:
            print "read_index : ",read_index
        read_index += 1

        line = line.strip().split()
        if len(line) > 1:
            tmp_x = np.zeros((max_doc_len), dtype=np.float32)
            # featurelist = line[1].split()
            featurelist = line
            # print featurelist
            i=0
            for feature in featurelist:
                # print feature
                if i >=max_doc_len:
                    break
                
                temp = feature.split(":")
                tmp_x[int(temp[0])] = float(temp[1])
                i+=1
            x.append(tmp_x)
        else:
            tmp_x = np.zeros((max_doc_len), dtype=np.int)
            x.append(tmp_x)

    # print "output TFIDF feature .."    

    # np.savetxt('TDIDF.txt', x, delimiter=' ')

    # output TFIDF feature
    # print "output TFIDF feature .."
    # fw = open('TFIDF.txt','w')
    # read_index = 0
    # for item in x:
    #     if read_index%100 == 0:
    #         print "read_index : ",read_index
    #     read_index += 1

    #     string = ""
    #     for it in item:
    #         string += str(it) + " "
    #     print >> fw,string

    # return x


    print "done!"



if __name__ == '__main__':
    input_file = 'train_and_test.sample'
    x = load_data(input_file, max_doc_len= 4000 + 1)
    print len(x)
    print x[0]