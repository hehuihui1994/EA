# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:59:30 2016

@author: huihui
"""

import performance
import copy

#词典由三部分组成，DUTIR、俚语词、情感符emoticons
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
            
 
#输入一个句子，输出一种情绪
def algorithm(lineSet,dic_DUTIR,dic_slang,dic_smile,dic_kafang):
    emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise']
    n=[0 for i in range(0,7)]
    for word in lineSet:
        if dic_DUTIR.has_key(word):
            for i in range(0,7):
                for e in dic_DUTIR[word]:
                    if e==emotion[i]:
                        n[i]=n[i]+1                        
        elif dic_slang.has_key(word):
            for i in range(0,7):
                for e in dic_slang[word]:
                    if e==emotion[i]:
                        n[i]=n[i]+1
        elif dic_smile.has_key(word):
            for i in range(0,7):
                for e in dic_smile[word]:
                    if e==emotion[i]:
#                        print e
                        n[i]=n[i]+1                        
        elif dic_kafang.has_key(word): 
#            print word
            for i in range(0,7):
                for e in dic_kafang[word]:
                    if e==emotion[i]:
                        n[i]=n[i]+1  
                    
    
    nTemp=sorted(n)
    for i in range(0,7):
        if n[i]==nTemp[6]:
            if n[i]!=0:
                return emotion[i]
            else:
                return 'none'

#整个测试集的句子级结果
def process(fileName,dic_kafang,dic_DUTIR,dic_slang,dic_smile):  
    result=[]
    f=open(fileName,'r')
    for line in f.readlines():
        lineSet=line.strip().split()
        sentence_result=algorithm(lineSet,dic_DUTIR,dic_slang,dic_smile,dic_kafang)
        result.append(sentence_result)	  
    return result
    
	
#读入人工标注的结果
def readin_label(filename):
    label=[]
    f=open(filename,'r')
    for line in f.readlines():
        labelLine=line.strip().split()
        label.append(labelLine[3])
    return label

def read_smile(filename):
    dic_smile={}
    f=open(filename,'r')
    for line in f.readlines():
        lineSet=line.strip().split()
        temp=[]
        temp.append(lineSet[1])
        dic_smile[lineSet[0]]=temp
    return dic_smile



if __name__ == '__main__':
    #dic_kafang=read_dic('x2')
    dic_DUTIR_old=read_dic('dutir_new')
    dic_slang_old=read_dic('slang_new')
    dic_smile_old=read_dic('smile_new')
    dic_kafang_old=read_dic('x2_new')
   
    result=process('sentence_e.txt',dic_kafang_old,dic_DUTIR_old,dic_slang_old,dic_smile_old)
  
    #输出到文件dic_result
    f=open('dic_result','w')
    for item in result:
        print>>f,item
   
   
  
    #输出各类指标
    print("sentence情绪识别任务------------")        
    #人工标注
    label=readin_label('sentence_test_label.txt') 
    # label=readin_label('weibo_label.txt')
    #正确率acc
    accuracy=performance.calc_acc(result,label)
    print("正确率：%r"%(accuracy))
 
    #宏平均
    class_dict1={'happiness':'happiness','like':'like','anger':'anger',
    'sadness':'sadness','fear':'fear','disgust':'disgust','surprise':'surprise'}
    macro_dict1=performance.calc_macro_average(result,label,class_dict1)    
    
    #每一类情绪
    class_dict={'happiness':'happiness','like':'like','anger':'anger',
    'sadness':'sadness','fear':'fear','disgust':'disgust','surprise':'surprise',
    'none':'none'}
    #precision
    precision_dict=performance.calc_precision(result,label,class_dict)
    print("macro_precision——%r"%(macro_dict1['macro_p']))
    for i in class_dict:
        print("%r:%r"%(class_dict[i],precision_dict[class_dict[i]]))
    #recall
    recall_dict=performance.calc_recall(result,label,class_dict)
    print("macro_recall——%r"%(macro_dict1['macro_r']))
    for i in class_dict:
        print("%r:%r"%(class_dict[i],recall_dict[class_dict[i]]))   
    #f-measure
    fscore_dict=performance.calc_fscore(result,label,class_dict)
    print("macro_fscore——%r"%(macro_dict1['macro_f1']))
    for i in class_dict:
        print("%r:%r"%(class_dict[i],fscore_dict[class_dict[i]]))
    print("-------------------------")

        