# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:01:30 2016

@author: huihui
"""

def train_to_emotion(fileName1,fileName2):
    #train_emotion_quzao.txt_fenci
    f1=open(fileName1,'r')
    train_text=[]
    for line in f1.readlines():
        lineSet=line.strip().split()
        train_text.append(lineSet)
    #train_emotion_label.txt
    emotion_label=[]
    f2=open(fileName2,'r')
    for line in f2.readlines():
        emotion_label.append(line.strip())
    emotion=['happiness','like','anger',
    'sadness','fear','disgust','surprise']
    dic_emotion_text={}
    for item in emotion:
        temp=[]
        for i in range(0,len(emotion_label)):
            if emotion_label[i]==item:
                temp.append(train_text[i])
        dic_emotion_text[item]=temp
    return dic_emotion_text
    
def x2(t,c,dic_emotion_text):
    M=len(dic_emotion_text[c])
    A=0
    for sentence in dic_emotion_text[c]:
        if t in sentence:
            A=A+1
    C=M-A
    B=0
    emotion=['happiness','like','anger',
    'sadness','fear','disgust','surprise']
    for item in emotion:
        if item!=c:
            for sentence in dic_emotion_text[item]:
                if t in sentence:
                    B=B+1
    N=2172
    D=N-A-B-C
    X=(A*D-B*C)*(A*D-B*C)*1.0/((A+B)*(C+D))
    return X
    
def x2_select():
    f=open('x2_word_3000.txt','w')
    dic_emotion_text=train_to_emotion("train_removeStopWord.txt","train_emotion_label.txt")
    emotion=['happiness','like','anger',
    'sadness','fear','disgust','surprise']
    #去除DUTIR中出现过的词
    dutir=[]
    f0=open('dic_intention_polarity.txt','r')
    for line in f0.readlines():
        lineSet=line.strip().split()
        dutir.append(lineSet[0])       
    word=[]
    f1=open("train_removeStopWord.txt",'r')
    for line in f1.readlines():
        lineSet=line.strip().split()
        for t in lineSet:
            if t in dutir:
                continue
            if t in word:
                continue
            word.append(t)
                         
    for c in emotion:
        score=[]
        for t in word:
            print t
            s=x2(t,c,dic_emotion_text)
            score.append(s)
            print("%r:%r"%(c,x2(t,c,dic_emotion_text)))
        #选取前100个词作为类别c的特征词
        scoreTemp=sorted(score)
        #由大到小排序
        scoreTemp.reverse()
        #排序的分数不能重复
        scoreTemp1=[]
        for x in scoreTemp:
            if x in scoreTemp1:
                continue
            scoreTemp1.append(x)
        for tempScore in scoreTemp1:
            if tempScore<3000:
                break
            for j in range(0,len(score)):
                if score[j]==tempScore:
                    string=word[j]+" "+c+" "+str(tempScore)
                    print >>f,string
                
            


#处理卡方生成的数据
def process():
    f1=open('x2_word_3000.txt','r')
    f2=open('x2_new_test','w')
    dic={'happiness':'PA','like':'PB','anger':'NA',
    'sadness':'NB','fear':'NI','disgust':'NE','surprise':'PC'}
    dic_emotion={}
    for line in f1.readlines():
        lineSet=line.strip().split()
        temp=[lineSet[1],lineSet[2]]
        if dic_emotion.has_key(lineSet[0]):
            if dic_emotion[lineSet[0]][1]<lineSet[2]:
                dic_emotion[lineSet[0]]=temp             
        else:
            dic_emotion[lineSet[0]]=temp
            #778
    score=[4000,6000,8000,10000,30000]
    num=['1','3','5','7','9']
    for item in dic_emotion:
        string=item+" "+dic[dic_emotion[item][0]]+" "
        for i in range(0,5):
            temp=float(dic_emotion[item][1])
            if temp <score[i]:
                string=string+num[i]+" "
                if dic_emotion[item][0]=="happiness" or dic_emotion[item][0]=="like":
                    string=string+"1"
                else:
                    string=string+"2"
                print>>f2,string
                break
        




if __name__=='__main__': 
#    x2_select()
    process()
    