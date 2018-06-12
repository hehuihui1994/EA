# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:05:03 2016

@author: huihui
"""

#单词特征+标点符号特征
def read_word_feature():
    feature=[]
    f=open('train_quzao.txt_fenci','r')
    for line in f.readlines():
        lineSet=line.strip('，|：|。|~| [ | ]').split()
        for word in lineSet:
            if word in feature:
                continue
            else:
                feature.append(word)
    return feature


#读入中文情绪词典
def readin_DUTIR():
    dic_DUTIR={}
    f=open("dic_DUTIR.txt",'r')
    for line in f.readlines():
        lineSet=line.strip().split()
        temp=[]
        for i in range(1,len(lineSet)):
            temp.append(lineSet[i])
        dic_DUTIR[lineSet[0]]=temp
    return dic_DUTIR
        
        
#处理数据变成LIBSVM接受的模式
def to_libsvm(fileName1,fileName2,feature_word_punc,dic_DUTIR):
     f2=open("train_label_int.txt",'r')  
     #标签是数字
     train_label=[]
     for line in f2.readlines():
        line=line.strip()
        train_label.append(line)
       
     f3=open(fileName1,'r')
     f4=open(fileName2,'w')
     #情绪类别
     emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise']
     #先dic_DUTIR（7维），再feature_word_punc
     index=0
     for line in f3.readlines():
        lineSet=line.strip('，|：|。|~| [ | ]').split()
        #string=train_label[index]+" "
        string="1 "
        n=[0 for i in range(0,7)]
        for word in lineSet:
            if dic_DUTIR.has_key(word):
                for i in range(0,7):
                    for e in dic_DUTIR[word]:
                        if e==emotion[i]:
                            n[i]=n[i]+1
        for i in range(0,7):
             if(n[i]==0):
                 continue             
             j='%d'%(i+1)
             num='%d'%(n[i])
             string=string+j+":"+num+" "
     
        index=index+1
        #词和拼音特征
        for i in range(0,len(feature_word_punc)):
            if feature_word_punc[i] in lineSet:
                 #统计词频
                 num=0
                 for word in lineSet:
                     if(word==feature_word_punc[i]):
                         num=num+1
                 j='%d'%(i+8)
                 num='%d'%(num)
                 string=string+j+":"+num+" "
        print>>f4,string
         
            
if __name__=='__main__':
    feature_word_punc=read_word_feature()
    dic_DUTIR=readin_DUTIR()
    #to_libsvm('train_quzao.txt_fenci','train_libsvm_all',feature_word_punc,dic_DUTIR)
    to_libsvm('test_quzao.txt_fenci','test_libsvm_all',feature_word_punc,dic_DUTIR)