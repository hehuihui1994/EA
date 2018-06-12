# -*- coding: utf-8 -*-
"""


@author: huihui
"""

import performance
import math
import copy

#情感词典由DUTIR和卡方检验获得的扩展情感词组成
def dic(fileName):
    dic_intention_polarity={}
    f=open(fileName,'r')
    for line in f.readlines():
        lineSet=line.strip().split()
        temp=[lineSet[1],lineSet[2],lineSet[3]]
        dic_intention_polarity[lineSet[0]]=temp
    return dic_intention_polarity



def for_weight(lineSet,dic_intention_polarity,dic_kafang,dic_smile,a,b):
    weight=[0 for i in range(21)]
    emotion_21=['PA','PE','PD','PH','PG','PB','PK','NA','NB','NJ','NH','PF','NI','NC','NG','NE','ND','NN','NK','NL','PC']
    for word in lineSet:
        if dic_intention_polarity.has_key(word):
            for i in range(0,21):
                if dic_intention_polarity[word][0]==emotion_21[i]:
                    #print dic_intention_polarity[word][0]
                    x=int(dic_intention_polarity[word][1])
                    y=int(dic_intention_polarity[word][2])
                    weight[i]=weight[i]+a*x+b*y
                    break
        
        if dic_kafang.has_key(word):
            for i in range(0,21):
                if dic_kafang[word][0]==emotion_21[i]:
                    #print dic_intention_polarity[word][0]
                    x=int(dic_kafang[word][1])
                    y=int(dic_kafang[word][2])
                    weight[i]=weight[i]+a*x+b*y
                    break
        
        if dic_smile.has_key(word):
            for i in range(0,21):
                if dic_smile[word][0]==emotion_21[i]:
                    #print dic_intention_polarity[word][0]
                    x=int(dic_smile[word][1])
                    y=int(dic_smile[word][2])
                    weight[i]=weight[i]+a*x+b*y
                    break
        
    return weight


#将训练集合中有情绪的微博表示为21维向量
def train_to_vector(dic_intention_polarity,dic_kafang,dic_smile,a,b):
    train_vec=[]
    # f=open('train_emotion_quzao.txt_fenci','r')
    f=open('train_emotion_e','r')
    for line in f.readlines():
        lineSet=line.strip().split()
        weight=for_weight(lineSet,dic_intention_polarity,dic_kafang,dic_smile,a,b)
        train_vec.append(weight)
    return train_vec


#将测试集合中有情绪的微博表示为21维向量
def test_to_vector(dic_intention_polarity,dic_kafang,dic_smile,a,b):
    f_yn=open('result_emotion_tool.txt','r')
    result=[]
    for line in f_yn.readlines():
        result.append(line.strip())
    index=0
    test_vec=[]
    # f_w=open('weibo_quzao.txt_fenci','r')
    f_w=open('weibo_e','r')
#    f_e=open('weibo_emotion','w')
    for line in f_w.readlines():
        if result[index]=='N':
            index=index+1
            continue
        index=index+1
#        print>>f_e,line.strip()
        lineSet=line.strip().split()
        weight=for_weight(lineSet,dic_intention_polarity,dic_kafang,dic_smile,a,b)
        test_vec.append(weight)
    return test_vec



#对测试集中的每一条微博，计算与训练集中每一条微博的向量相似度
#k=21
def KNN(a,b,dic_intention_polarity,dic_kafang,dic_smile):
    # fw=open('test_result_temp','w')
    #训练集label
    train_label=[]
    f_train_label=open('train_emotion_label.txt','r')
    for line in f_train_label.readlines():
        train_label.append(line.strip())
    #训练vec
    train_vec=train_to_vector(dic_intention_polarity,dic_kafang,dic_smile,a,b)
    #测试vec
    test_vec=test_to_vector(dic_intention_polarity,dic_kafang,dic_smile,a,b)
    
    #预测的结果
    test_emotion=[]
    for test_weight in test_vec:
        #相似度值
        cos=[]
        for train_weight in train_vec:
            up=0.0
            down1=0.0
            down2=0.0
            for i in range(0,len(train_weight)):
                up=up+train_weight[i]*test_weight[i]
                down1=down1+train_weight[i]*train_weight[i]
                down2=down2+test_weight[i]*test_weight[i]
            down1=math.sqrt(down1)
            down2=math.sqrt(down2)
            if down1==0 or down2==0:
                cos_theta=-2
                cos.append(cos_theta)
                continue
            #print("train %r"%(down1))
            #print("test %r"%(down2))
            down=down1*down2
            cos_theta=up*1.0/down
            cos.append(cos_theta)
        #降序排序，取前21个相似度
        cosTemp=sorted(cos)
        cosTemp.reverse()
        train_index=[]
        for i in range(0,21):
            for j in range(0,len(cos)):
                if cos[j]==cosTemp[i]:
                    train_index.append(j)
                    cos[j]=2
                    break
        #计算结果train_label
        emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise']
        n=[0 for i in range(0,7)]
        for i in train_index:
            for j in range(0,len(emotion)):
                if train_label[i]==emotion[j]:
                    n[j]=n[j]+1
        #选出最大的
        max=-1
        test_emotion_temp=""
        for i in range(0,7):
            if n[i]>max:
                max=n[i]
                test_emotion_temp=emotion[i]
        test_emotion.append(test_emotion_temp)
        # print>>fw,test_emotion_temp
#    print len(test_emotion)
    return test_emotion
    



    
#合并情绪判断与情绪识别结果,也就是预测的结果
def merge_result(fileName2,emotion_label):
    #test_result_temp
    # f1=open(fileName1,'r')
    #result_emotion
    f2=open(fileName2,'r')
    f_out=open('result_old.txt','w')
    #记录情绪识别结果
    # emotion_label=[]
    # for line in f1.readlines():
    #     emotion_label.append(line.strip())
    test_out=[]
    index=0
    for line in f2.readlines():
        if line.strip()=='N':
            temp='none'
            test_out.append(temp)
            print>>f_out,temp
        else:
            temp=emotion_label[index]
            index=index+1
            test_out.append(temp)
            print>>f_out,temp
    return test_out


def score_emotion(fileName1,fileName2):
    #标注结果
    label=[]
    f=open(fileName1,'r')
    for line in f.readlines():
        labelLine=line.strip().split()
        label.append(labelLine[2])
    #预测结果
    result=[]
    f1=open(fileName2,'r')
    for line in f1.readlines():
        result.append(line.strip())
    #输出各类指标
    print("weibo情绪识别任务------------")        
 
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



if __name__ == '__main__':
    #人工标注标签
    # label=[]
    # f_label=open('label.txt','r')
    # for line in f_label.readlines():
    #     lineSet=line.strip().split()
    #     label.append(lineSet[2])
    # #
    # dic_intention_polarity=dic('dic_intention_polarity.txt')
    # dic_kafang=dic('kafang3000')
    # #smile
    # dic_smile=dic('smile_new_new.txt')
    # test_emotion_old=KNN(0.9,0.1,dic_intention_polarity,dic_kafang,dic_smile)
    # result_old= merge_result("result_emotion_bo.txt",test_emotion_old)
    # score_emotion('label.txt',result_old) 
    #输出结果
    score_emotion('label.txt','result_old.txt')