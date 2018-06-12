#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 2017

@author: hehuihui1994 (he__huihui@163.com)
https://github.com/hehuihui1994
"""

# 24  + 8
# the number of words in the tweet matching each class are counted 8
# the individual scores for each class are summed 8
# bool for each class 8
# npmi 8

import numpy as np

def read_dic(fileName):
    dic={}
    f=open(fileName,'r')
    for line in f.readlines():
        lineSet=line.strip().split()
        temp=[]
        for i in range(1,len(lineSet)):
            temp.append(lineSet[i])
        dic[lineSet[0]]=temp
    return dic

def read_npmi_dic(fileName):
    dic = {}
    f = open(fileName, 'r')
    for line in f.readlines():
        line = line.strip().split()
        tmp = []
        for i in range(1,len(line)):
            tmp.append(float(line[i]))
        dic[line[0]] = np.asarray(tmp)
    return dic
                

def get_seed():
    fr = open('../../resources/renCECps_text.txt','r')
    fw = open('../../resources/dutir_ren.txt','w')
    emotions = ['Joy', 'Hate', 'Love', 'Sorrow',
              'Anxiety', 'Surprise', 'Anger', 'Expect']
    #  read dic_dutir
    dic_dutir = read_dic('../../resources/dutir.txt')
    # find word in dutir
    wordlist = []
    for line in fr.readlines():
        if len(line) == 0:
            continue
        else:
            line = line.strip().split()
            for word in line:
                if dic_dutir.has_key(word): 
                    if word in wordlist:
                        continue
                    else:
                        wordlist.append(word)
    for w in wordlist:
        string = w+ " "
        for it in dic_dutir[w]:
            string += it + " "
        # print"{} {}".format(w, dic_dutir[w])
        print >> fw, string
    print len(wordlist)    


def get_dic_feature():
    fw = open('dic32_chi.txt','w')
    fr = open('../../resources/renCECps_text.txt','r')
    emotions = ['Joy', 'Hate', 'Love', 'Sorrow',
              'Anxiety', 'Surprise', 'Anger', 'Expect']
    #  read dic_dutir
    dic_dutir = read_dic('../../resources/dutir.txt')

    # read npmi dic
    dic_npmi = read_npmi_dic('../../resources/secp_prec_npmi_dic_df4_chi.txt')

    #  get dic feature
    for line in fr.readlines():
        string = ""
        if len(line) == 0:
            print>>fw,"0."
        else:
            line = line.strip().split()
            tmp_dic = [{} for i in range(3)]
            dic1 = {}
            dic2 = {}
            dic3 = {}
            for e in emotions:
                dic1[e] = 0.
                dic2[e] = 0.
                dic3[e] = 0.

            sum_dic_npmi = np.asarray([0.0 for i in range(8)])
            num_npmi = 0.

            for word in line:
                if dic_dutir.has_key(word):   
                    item = dic_dutir[word]
                    if len(item) == 2:
                        dic1[item[0]] += 1.
                        dic2[item[0]] += float(item[1])
                        dic3[item[0]] = 1.
                    elif len(item) == 4:
                        dic1[item[0]] += 1.
                        dic2[item[0]] += float(item[1])
                        dic3[item[0]] = 1.
                        dic1[item[2]] += 1.
                        dic2[item[2]] += float(item[3])
                        dic3[item[2]] = 1. 
                elif dic_npmi.has_key(word):
                    num_npmi += 1
                    sum_dic_npmi += dic_npmi[word]
            # print sum_dic_npmi
            # print num_npmi
            if sum(sum_dic_npmi) != 0.0:
                sum_dic_npmi = sum_dic_npmi/num_npmi  
            # print sum_dic_npmi       
            
            for e in emotions:
                string += str(dic1[e]) + " "
            for e in emotions:
                string += str(dic2[e]) + " "               
            for e in emotions:
                string += str(dic3[e]) + " "    

            for it in sum_dic_npmi:
                string += str(it) + " "
            # print string
            # break  
            print >> fw,string                               
                        



    

if __name__ == '__main__':
    get_dic_feature()
    # get_seed()


