# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:03:48 2016

@author: huihui
"""
import  xml.dom.minidom
import sys

reload(sys)
sys.setdefaultencoding('utf8')
            
   
def train_label():
    #打开xml文档
    dom = xml.dom.minidom.parse('train.xml')  
    #得到文档元素对象
    root = dom.documentElement  
   #获得标签属性值
    itemlist0 = root.getElementsByTagName('weibo')
    #print  itemlist0[0].firstChild.data
    
    #输出文件
    f4=open('train_label.txt','w')
    for weibo in itemlist0:
        for sentence in weibo.getElementsByTagName('sentence'):
            string=weibo.getAttribute('id')+" "+weibo.getAttribute('emotion-type')+" "
            if sentence.getAttribute('emotion_tag')=='N':
                string=string+sentence.getAttribute('id')+" "+"none"+" "+"none"
            else:
                string=string+sentence.getAttribute('id')+" "+sentence.getAttribute('emotion-1-type')+" "+sentence.getAttribute('emotion-2-type')+" "
            print>>f4,string

def get_train_Y():
    #打开xml文档
    dom = xml.dom.minidom.parse('train.xml')  
    #得到文档元素对象
    root = dom.documentElement  
   #获得标签属性值
    itemlist0 = root.getElementsByTagName('weibo')
    f4=open('train_Y.txt','w')
    for weibo in itemlist0:
        print>>f4,weibo.getAttribute('emotion-type')

        

def get_conjunctions():
       f=open('conjunction_words.txt','r')
       conjunctions=[]
       for line in f.readlines():
              lineSet=line.strip().split()
              for word in lineSet:
                     if word in conjunctions:
                            continue
                     else:
                            conjunctions.append(word)
       #print len(conjunctions)
       return conjunctions

#只要人工标注的主要情绪
def get_train_X():
    fw=open('train_X.txt','w')
    conjunctions=get_conjunctions()
    fr_label=open('train_label.txt','r')
    fr_sentence=open('sentence_train_quzao.txt_fenci','r')
    #<'none'>
    csr_label=[]
    temp=[]
    index=1
    for line in fr_label.readlines():
        lineSet=line.strip().split()
        if int(lineSet[0])==index:
            temp.append(lineSet[3])
        else:
            csr_label.append(temp)
            index=index+1
            temp=[]
            temp.append(lineSet[3])
    #第4000个
    csr_label.append(temp)
    #读入train的句子
    sentence=[]
    for line in fr_sentence.readlines():
        lineSet=line.strip().split()
        sentence.append(lineSet)
    #train_csr_x
    num=0
    for label in csr_label:
        string=""
        for i in range(num,num+len(label)):
             if len(sentence[i])!=0:
                if sentence[i][0] in conjunctions:
                    string=string+sentence[i][0]+" "+label[i-num]+" "
                else:
                    string=string+label[i-num]+" " 
             else:
                string=string+label[i-num]+" "   
        print>>fw,string
        num=num+len(label)



def get_test_X():
    fw=open('test_X_dic.txt','w')
    conjunctions=get_conjunctions()
    test_ID=[]
    fr_test_ID=open('test_ID.txt','r')
    for line in fr_test_ID.readlines():
        lineSet=line.strip().split()
        test_ID.append(lineSet[0])
    dic_result=[]
    fr_dic=open('dic_result_final','r')
    for line in fr_dic.readlines():
        lineSet=line.strip().split()
        dic_result.append(lineSet[0])
    svm_result=[]
    fr_svm=open('dic_result_final','r')
    for line in fr_svm.readlines():
        lineSet=line.strip().split()
        svm_result.append(lineSet[0])
    ##< <disgust,none> , <none> >
    index=1
    temp=[]
    tempSameSentence=[]
    csr_label=[]
    for i in range(0,len(test_ID)):
        if int(test_ID[i])==index:
            if dic_result[i]==svm_result[i]:
                tempSameSentence=[dic_result[i]]
                temp.append(tempSameSentence)
            else:
                tempSameSentence=[dic_result[i],svm_result[i]]
                temp.append(tempSameSentence)
        else:
            csr_label.append(temp)
            index=index+1
            temp=[]
            if dic_result[i]==svm_result[i]:
                tempSameSentence=[dic_result[i]]
                temp.append(tempSameSentence)
            else:
                tempSameSentence=[dic_result[i],svm_result[i]]
                temp.append(tempSameSentence)
    csr_label.append(temp)
    #加入连接词
    fr_sentence=open('sentence_test_quzao.txt_fenci','r')
    sentence=[]
    for line in fr_sentence.readlines():
        lineSet=line.strip().split()
        sentence.append(lineSet)
    #test_csr_x
    num=0
    for label in csr_label:
        string=""
        for i in range(num,num+len(label)):
            if len(sentence[i])!=0:
                if sentence[i][0] in conjunctions:
                    if len(label[i-num])!=1:
                        string=string+sentence[i][0]+" "+label[i-num][0]+","+label[i-num][1]+" "
                    else:
                        string=string+sentence[i][0]+" "+label[i-num][0]+" "
                else:
                    if len(label[i-num])!=1:
                        string=string+label[i-num][0]+","+label[i-num][1]+" "
                    else:
                        string=string+label[i-num][0]+" " 
            else:
                if len(label[i-num])!=1:
                    string=string+label[i-num][0]+","+label[i-num][1]+" "
                else:
                    string=string+label[i-num][0]+" "       
        print>>fw,string
        num=num+len(label)


#######################################################################################
#从train中获取si
def get_sequence_from_file(fileName):
    fr=open(fileName,'r')
    csr_x=[]
    for line in fr.readlines():
        temp=[]
        tempSameSentence=[]
        lineSet=line.strip().split()
        for word in lineSet:
            for item in word.split(','):
                tempSameSentence.append(item)
            temp.append(tempSameSentence)
            tempSameSentence=[]
        csr_x.append(temp)
    return csr_x


#求一个序列的所有子序列
def get_sub_sequence(csr_train_X):
    fw=open('sub_train_x','w')
    sub=[]
    index1=0
    for sequence in csr_train_X:
        # print sequence
        print index1
        index1+=1
        seq_len=len(sequence)
        num=2**seq_len
        bi_num=[]
        for i in range(0,num):
            bi_num.append(bin(i).replace('0b',''))
        for item in bi_num:
            len_item=len(item)
            temp=[]
            index=0
            for i in range((seq_len-len_item),seq_len): 
                if item[index]=='1':
                    temp.append(sequence[i][0])
                index+=1

            if (temp in sub) or (len(temp)==0) :
                continue
            else:
                sub.append(temp)
                string=""
                for e in temp:
                    string+=e+" "
                print string
                print>>fw,string


if __name__ == '__main__':
    class_dict={'happiness':'1','like':'2','anger':'3',
    'sadness':'4','fear':'5','disgust':'6','surprise':'7',
    'none':'8'}
    # train_label()
    # get_train_Y()
    # get_train_X()
    # get_test_X()
    #从训练集合中提取CSR
    #Si的所有子序列
    # fw_train_x=open('train_x_quchong','w')
   #  fw_tarin_y=open('train_y_quchong_emotion','w')
   #  csr_train_X_temp=get_sequence_from_file('train_X.txt')
   #  train_Y_temp=[]
   #  f=open('train_Y.txt','r')
   #  for line in f.readlines():
   #     lineSet=line.strip()
   #     train_Y_temp.append(lineSet)
   #  #train中去掉重复的(Si,Yi)
   #  csr_train_X=[] 
   #  train_Y=[]

   #  csr_train_X.append(csr_train_X_temp[0])
   #  train_Y.append(train_Y_temp[0])
   #  for i in range(1,len(train_Y_temp)):
   #     flag=0
   #     for j in range(0,len(train_Y)):
   #         if csr_train_X_temp[i]==csr_train_X[j] and train_Y_temp[i]==train_Y[j]:
   #             flag=1
   #             break
   #     if flag==1:
   #         continue
   #     else:
   #         csr_train_X.append(csr_train_X_temp[i])
   #         train_Y.append(train_Y_temp[i])
   # #csr_train_X,train_Y为不重复的（si,yi）
   # #写出到文件
   #  # for csr_x in csr_train_X:
   #  #     string=""
   #  #     for x in csr_x:
   #  #         string+=x[0]+" "
   #  #     print>>fw_train_x,string

   #  for label in train_Y:
   #      print>>fw_tarin_y,label




   # #获取csr_train_X的所有子序列1289，去重后，写入到sub_train_x
   #  print len(csr_train_X)
   # print csr_train_X[528]
   # string=""
   # for items in csr_train_X[528]:
   #     string+=items[0]+" "
   # print string 

    ###训练集中的所有子序列###
    # get_sub_sequence(csr_train_X)