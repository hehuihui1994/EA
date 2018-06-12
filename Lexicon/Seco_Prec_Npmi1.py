#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 2017

@author: hehuihui1994 (he__huihui@163.com)
https://github.com/hehuihui1994
"""

import numpy as np
import math
import pytc


class Seco_Prec_NPMI(object):
    
    def __init__(self, dic_name, corpus_name, corpus_labels, final_dic_name, stopWords_Name, k, df):
        self.dic_name = dic_name
        self.corpus_name = corpus_name
        self.corpus_labels = corpus_labels
        self.final_dic_name = final_dic_name
        self.stopWords_Name = stopWords_Name
        self.k = k,
        self.df = df

    #read seed
    def read_seed(self):
        dic_seed={}
        seed_all=[]
        e=[[] for i in range(0,8)]
        emotion = ['Joy', 'Hate', 'Love', 'Sorrow',
                'Anxiety', 'Surprise', 'Anger', 'Expect']
        f=open(self.dic_name,'r')
        for line in f.readlines():
            lineSet=line.strip().split()
            seed_all.append(lineSet[0])
            for i in range(0,len(emotion)):
                if lineSet[1]==emotion[i]:
                    e[i].append(lineSet[0])
                if len(lineSet) > 3:
                    if lineSet[3]==emotion[i]:
                        e[i].append(lineSet[0])                            
        for i in range(0,len(emotion)):
            dic_seed[emotion[i]]=set(e[i])
        # seed_all = np.asarray(seed_all)
        self.dic_seed = dic_seed
        self.seed_all = set(seed_all)
        # return dic_seed, set(seed_all)   

    def get_M(self):
        fr = open(self.corpus_name,'r')
        sentences = []
        wordList = []
        for line in fr.readlines():
            sentences.append(line)
            line = line.strip().split()
            wordList.extend(line)

        doc_str_list_train = sentences

        labels = []
        f_label = open(self.corpus_labels, 'r')
        for line in f_label.readlines():
            l = line.strip()
            labels.append(l)   
        doc_class_list_train = labels       

        doc_terms_list_train= pytc.get_doc_terms_list(doc_str_list_train)

        class_set = pytc.get_class_set(doc_class_list_train)
        #class_set
        for i in range(0,len(class_set)):
            string=str(i)+" "+class_set[i]
            print string

        term_set = pytc.get_term_set(doc_terms_list_train)
        print"    number of terms",len(term_set)

        print u'    feature filter (DF>=4)...' 
        #   default >4
        term_df = pytc.stat_df_term(term_set, doc_terms_list_train)
        term_set_df = pytc.feature_selection_df(term_df, self.df)
        term_set = term_set_df
        print"    number of terms after filter ",len(term_set)

        candiList = term_set

        # rmv seed word to get candidate
        # wordList = np.asarray(wordList)
        candiList =  list(set(candiList).difference(set(self.seed_all)))
        print "    read stop words..."
        fr_stop = open(self.stopWords_Name, 'r')
        stopWords = [line.strip() for line in fr_stop.readlines()]
        print "    num of stop words",len(stopWords)

        candiList = list(set(candiList).difference(set(stopWords)))
        print "    number of terms after rmv stopwords",len(candiList) 

        print u'feature select chi...'
        df_class = pytc.stat_df_class(class_set, doc_class_list_train)
        df_term_class = pytc.stat_df_term_class(candiList, class_set, \
            doc_terms_list_train, doc_class_list_train)
        
        fs_method='CHI'
        fs_num=0
        term_set_fs, term_score_dict = pytc.supervised_feature_selection(df_class, \
            df_term_class, fs_method, fs_num)
        candiList = term_set_fs

        candiList = candiList[:4000]
        

        len_term_set = len(candiList)
        print(u"number of feature after CHI ：%r"%(len_term_set))

        
        print " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"       
        M = len(wordList)
        print "    num of words in corpus M = ",M
        print "    num of differ words in corpus : ", len(set(wordList))
        print "    num of seed word : ", len(set(self.seed_all))
        print "    num of candidate words : ",len(candiList)
        self.M = M
        self.candiList = candiList
        # return M, candiList 

    #compute #(w,t) #(w) #(t) within a context window
    # the closest preceding seed word to the cue word
    def get_frequence_within_window(self, tmp_line):
        # string = ""
        # for it in tmp_line:
        #     string += it + " "
        # print "        ",string
        # print "        update dic_w  dic_t.. use ", string
                
        wList = []
        tList = []
        for word in list(set(tmp_line)):
            if word in self.candiList:
                self.dic_w[word] += 1
                wList.append(word)
            elif word in self.seed_all:
                self.dic_t[word] += 1
                tList.append(word)

        # print "        update dic_wt.. use ",string
        if len(tList) != 0:
            if len(tList) == 1:
                #  only one seed word
                t = tList[0]
                for w in wList:
                    string = w +t
                    self.dic_wt[string] += 1
            else:
                # more than one seed word
                # find the closest preceding seed word t
                # (1) preceding
                # (2) closest
                for w in wList:
                    indexW = tmp_line.index(w)
                    tmpT = tList[0]
                    for t in tList:
                        indexT = tmp_line.index(t) 
                        indexTmpT = tmp_line.index(tmpT)
                        #  indexTmpT X indexT
                        if indexW > indexTmpT and indexW < indexT:
                            continue
                        elif indexW > indexT and indexW < indexTmpT:
                            # indexT X indexTmpT
                            tmpT = t
                        elif abs(indexW - indexTmpT) < abs(indexW - indexT):
                            continue
                        else:
                            tmpT = t
                    string = w + tmpT
                    self.dic_wt[string] += 1
        # print "        finish update dic_wt use ",string
                            
                        
                        

    #  compute #(w,t) #(w) #(t) within a corpus
    def get_frequence(self):
        self.dic_w = {}
        self.dic_t = {}
        self.dic_wt = {}
        # self.distance_wt = {}

        # Initialization
        print "    initial dic_w dic_t dic_wt  ..."
        for w in self.candiList:
            self.dic_w[w] = 0.
        print "    finish initial dic_w.."
        for t in self.seed_all:
            self.dic_t[t] = 0.
        print "    finish initial dic_t.."           
        for w in self.candiList:
            for t in self.seed_all:
                string = w + t
                self.dic_wt[string] = 0.
                # self.distance_wt[string] = 0. 
        print "    finish initial dic_wt.."           
        
        print"    finish initial ..."
        
        print "    get frequence from corpus ",self.corpus_name,"..."
        fr = open(self.corpus_name,'r')
        window_index = 0.
        for line in fr.readlines():
            line = line.strip().split()
            #  context window k
            # if len(line) < k:
            #     tmp_line = line
            #     self.get_frequence_within_window(tmp_line)
            # else:
            #     for i in range(len(line)- k + 1):
            #         tmp_line = line[i: i+k]
            #         self.get_frequence_within_window(tmp_line)

            #  M words M windows
            for i in range(len(line)):
                start = i
                # print self.k[0]
                end = i + self.k[0]

                if end > len(line):
                    end = len(line)
                tmp_line = line[start: end]
                self.get_frequence_within_window(tmp_line)
                if window_index%1000 == 0:
                    print "    window index : ",window_index
                window_index += 1
                
        print "    finish get frquence from corpus ..."

        # return dic_t, dic_w, dic_wt
        # self.dic_w = dic_w
        # self.dic_t = dic_t
        # self.dic_wt = dic_wt
        # self.distance_wt = distance_wt



    # compute seco-npmi(w to t)
    def get_w2t(self, w, t):
        # NPMI
        string = w + t
        if self.dic_wt[string] > 0. :
            up = math.log(float(self.M * self.dic_wt[string])/(self.dic_w[w]*self.dic_t[t]))
            down = math.log(float(self.M)/self.dic_wt[string])
            npmi = up/down
            # result = npmi*(self.k - self.distance_wt(string) + 1)/self.k
            result= npmi
            return result
        else:
            return 0.


    # compute seco-npmi(w to e) 
    def get_w2e(self, w):
        w2e = []
        emotion = ['Joy', 'Hate', 'Love', 'Sorrow',
                'Anxiety', 'Surprise', 'Anger', 'Expect']
        for i in range(0,len(emotion)):
            #  seed word for emotion i 
            e_seed = self.dic_seed[emotion[i]]
            m = len(e_seed)
            sum_tmp = 0.
            for t in e_seed:
                w2t = self.get_w2t(w, t)
                sum_tmp += w2t
            tmp = sum_tmp/m
            w2e.append(tmp)
        return w2e

                

    # compute all seed words and output to file secp_prec_npmi_dic.txt
    def get_final_dic(self):
        fw = open(self.final_dic_name,'w')
        for w in self.candiList:
            w2e = self.get_w2e(w)
            # emotion score = 0 
            if sum(w2e) == 0.0:
                continue
            string = w + " "
            for item in w2e:
                string += str(item) + " "
            # print string
            print >> fw,string
        print "finish construct secp_prec_npmi_dic.txt !!"

    
    def run(self):
        print "setp 1 : get seed word from dutir ..."
        # dic_seed, seed_all = self.read_seed()
        self.read_seed()
        print "step 2 : get M and candidate words ... "
        # M, candiList = self.get_M()
        self.get_M()
        print "setp 3 : get  #(w,t) #(w) #(t) within a contex window ",self.k, "..."
        self.get_frequence()
        print "step 4 : get final dic to secp_prec_npmi_dic.txt"
        self.get_final_dic()
        print "Congratulations !!"


if __name__ == '__main__':
    obj = Seco_Prec_NPMI(
        dic_name = 'dutir_ren.txt',
        corpus_name = 'renCECps_text.txt',
        corpus_labels = 'renCECps_label.txt',
        final_dic_name = 'secp_prec_npmi_dic_df4_chi.txt',
        stopWords_Name = 'stopWords.txt',
        # context window
        k = 10,
        df = 4,
    )
    obj.run()
