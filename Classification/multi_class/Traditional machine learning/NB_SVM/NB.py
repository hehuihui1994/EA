# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:46:02 2016

@author: huihui
"""
import performance

def get_WordFreq(fileName):
    #计算出这些词的词频
    f1=open(fileName,'r')
    all_word={}
    for line in f1.readlines():
        lineSet=line.strip().split()
        for word in lineSet:
            if(all_word.has_key(word)):
                all_word[word]=all_word[word]+1
            else:
                all_word[word]=1
    return  all_word    



#从词考虑特征选择
def featureSelect():
    FeatureSet=[]
    #统计每个词在4000条训练集的有无情绪两类中出现的概率
    emotion_WordFreq=get_WordFreq("train_emotion_e.txt")
    none_emotion_WordFreq=get_WordFreq("train_none_emotion_e.txt")
    RemainWordFreq=[]
    #开始找特征，先从有情绪的文本中提取
    for word_emotion in emotion_WordFreq:
        if(none_emotion_WordFreq.has_key(word_emotion)):
            if(emotion_WordFreq[word_emotion]/none_emotion_WordFreq[word_emotion]>4 and emotion_WordFreq[word_emotion]>2):
                FeatureSet.append(word_emotion)
            else:
                RemainWordFreq.append(word_emotion)
        elif(emotion_WordFreq[word_emotion]>2):
             FeatureSet.append(word_emotion)
    #无情绪文本中提取
    for word_none_emotion in none_emotion_WordFreq:
        if(emotion_WordFreq.has_key(word_none_emotion)):
            if(none_emotion_WordFreq[word_none_emotion]/emotion_WordFreq[word_none_emotion]>4 and none_emotion_WordFreq[word_none_emotion]>2):
                FeatureSet.append(word_none_emotion)
            else:
                RemainWordFreq.append(word_none_emotion)
        elif(none_emotion_WordFreq[word_none_emotion]>2):
             FeatureSet.append(word_none_emotion)
    print len(FeatureSet)
    #检察特征不重复
    # featureSet_temp=[]
    # for w in FeatureSet:
    #     if w in featureSet_temp:
    #         continue
    #     else:
    #         featureSet_temp.append(w)
    # print len(featureSet_temp)
    return FeatureSet
        
   

#多项式模型
def NB(fileName):
    #选取特征集合
    featureSet=featureSelect()
    emotion_WordFreq=get_WordFreq("train_emotion_e.txt")
    none_emotion_WordFreq=get_WordFreq("train_none_emotion_e.txt")
    #有无情绪的概率
    p_emotion=2172.0/4000
    p_none=1828.0/4000 
    #特征
    v=len(featureSet)
    #每个特征属性分别对有无情绪两个类别的条件概率
    p={}
    sum0=0
    sum1=0
    for feature in featureSet:
        if(none_emotion_WordFreq.has_key(feature)):
            count0=none_emotion_WordFreq[feature]
            sum0=sum0+count0
        else:
            count0=0
        if(emotion_WordFreq.has_key(feature)):
            count1=emotion_WordFreq[feature]
            sum1=sum1+count1
        else:
            count1=0
        temp=[count0,count1]
        p[feature]=temp
    for feature in featureSet:
        p[feature][0]=(p[feature][0]+1)*1.0/(sum0+v)
        p[feature][1]=(p[feature][1]+1)*1.0/(sum1+v)
    #对于测试集计算有无情绪
    result=[]
    f=open(fileName,'r')
    for line in f.readlines():
        lineSet=line.strip().split()
        p0=p_none
        p1=p_emotion
        for word in lineSet:
            if word in featureSet:
                p0=p0*p[word][0]
                p1=p1*p[word][1]
        if(p0>p1):
            temp='N'
            result.append(temp)
        else:
            temp='Y'
            result.append(temp)
    return result

#计算出每个特征在某个类别中的文档频率
def get_DocFreq(featureSet,fileName):
    all_word={}
    f=open(fileName,'r')
    for line in f.readlines():
        lineSet=line.strip().split()
        for feature in featureSet:
            if feature in lineSet:
                if all_word.has_key(feature):
                    all_word[feature]+=1
                else:
                    all_word[feature]=1
    return all_word



#伯努利模型
def NB1(fileName):
    # fw1=open('nb_pro','w')
    #选取特征集合
    featureSet=featureSelect()
    # print('featureSet[0]： %r'%(featureSet[0]))
    emotion_DocFreq=get_DocFreq(featureSet,"train_emotion_e.txt")
    none_emotion_DocFreq=get_DocFreq(featureSet,"train_none_emotion_e.txt")
    # print("emotion_DocFreq : %r"%(emotion_DocFreq[featureSet[2]]) )
    # print("none_emotion_DocFreq : %r"%(none_emotion_DocFreq[featureSet[2]]) )
    #有无情绪的概率
    p_emotion=2172.0/4000
    p_none=1828.0/4000 
    # print>>fw1,str(p_emotion)+" "+str(p_none)
    #平滑
    v=2
    #每个特征属性分别对有无情绪两个类别的条件概率
    p={}
    for feature in featureSet:
        if(none_emotion_DocFreq.has_key(feature)):
            count0=none_emotion_DocFreq[feature]
        else:
            count0=0

        if(emotion_DocFreq.has_key(feature)):
            count1=emotion_DocFreq[feature]
        else:
            count1=0
        temp=[count0,count1]
        p[feature]=temp
    for feature in featureSet:
        p[feature][0]=(p[feature][0]+1)*1.0/(1828+v)
        p[feature][1]=(p[feature][1]+1)*1.0/(2172+v)
        #保留有效位数6位
        # print>>fw1,str('%.6g'%(p[feature][1]))+" "+str('%.6g'%(p[feature][0]))
    # #对于测试集计算有无情绪
    result=[]
    # fw=open('nb_result_hhh','w')
    f=open(fileName,'r')
    for line in f.readlines():
        lineSet=line.strip().split()
        p0=p_none
        p1=p_emotion
        for feature in featureSet:
            if feature in lineSet:
                p0=p0*p[feature][0]
                p1=p1*p[feature][1]
            else:
                p0=p0*(1-p[feature][0])
                p1=p1*(1-p[feature][1])

    #     #判断类别
        if(p0>p1):
            temp='N'
            result.append(temp)
        else:
            temp='Y'
            result.append(temp)
        # string="1:"+str(p1/(p0+p1))+" "+"2:"+str(p0/(p0+p1))
        # print>>fw,string
    return result


    
#情绪判断任务指标
def score(result):
    #人工标注
    label=[]
    f1=open('label.txt','r')
    for line in f1.readlines():
        lineSet=line.strip().split()
        label.append(lineSet[1])   
    #输出各类指标
    class_dict={'Y':'Y'}
    print("weibo情绪判断任务------------") 
    precision_dict=performance.calc_precision(result,label,class_dict)      
    print("precision:%r"%(precision_dict['Y']) )
    recall_dict=performance.calc_recall(result,label,class_dict)   
    print("recall:%r"%(recall_dict['Y']) )
    fscore_dict=performance.calc_fscore(result,label,class_dict)
    print("f1:%r"%(fscore_dict['Y']))   
    print("-------------------------")

#情感词典由DUTIR和卡方检验获得的扩展情感词组成
def dic():
    dic_intention_polarity={}
#    f=open('dic_intention_polarity.txt','r')
    f=open('kafang1','r')
    for line in f.readlines():
        lineSet=line.strip().split()
        temp=[lineSet[1],lineSet[2],lineSet[3]]
        dic_intention_polarity[lineSet[0]]=temp
    return dic_intention_polarity



def for_weight(lineSet,dic_intention_polarity,a,b):
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
    return weight

#选取训练集4000条微博中的2712条标注为有情绪的微博作为SVM分类模型的训练集合
def train_for_libsvm(fileName,fileName1,dic_intention_polarity,a,b):
    f=open(fileName,'r')
    f1=open(fileName1,'w')
    #读入有情绪得的微博的情绪标签
    label=[]
    dic={'happiness':1,'like':2,'anger':3,
    'sadness':4,'fear':5,'disgust':6,'surprise':7}
    f2=open('train_emotion_label.txt','r')
    for line in f2.readlines():
        lineSet=line.strip()
        label.append(dic[lineSet])
    
    index=0
    for line in f.readlines():
        lineSet=line.strip().split()
        string='%d '%(label[index])
        index=index+1
        weight=for_weight(lineSet,dic_intention_polarity,a,b)
        for i in range(0,21):
            if weight[i]==0:
                continue
            #print weight[i]
            #j='%d'%(i+1)
            j=str(i+1)
            num=str(weight[i])
            string=string+j+":"+num+" "
        print>>f1,string

#测试集10000处理为libsvm接受的形式
def test_for_libsvm(fileName,result,fileName1,dic_intention_polarity,a,b):
    f=open(fileName,'r')
    f1=open(fileName1,'w')    
    index=0
    for line in f.readlines():
        if result[index]=='N':
            index=index+1
            continue       
        lineSet=line.strip().split()
        string='1 '
        index=index+1
        weight=for_weight(lineSet,dic_intention_polarity,a,b)
        for i in range(0,21):
            if weight[i]==0:
                continue
            #print weight[i]
            #j='%d'%(i+1)
            j=str(i+1)
            num=str(weight[i])
            string=string+j+":"+num+" "
        print>>f1,string


#处理成NB接受的形式
def train_for_nb(file_in,file_out):
    featureSet=featureSelect()
    train_label=[]
    f_label=open('train_label_int.txt','r')
    for line in f_label.readlines():
        if line.strip()=='8':
            train_label.append('2')
        else:
            train_label.append('1')

    f=open(file_in,'r')
    fw=open(file_out,'w')
    index=0
    for line in f.readlines():
        lineSet=line.strip().split()
        string=train_label[index]+"\t"
        for i in range(0,len(featureSet)):
            if featureSet[i] in lineSet:
                string+=str(i+1)+":1 "
        print>>fw,string
        index+=1

def test_for_nb(file_in,file_out):
    featureSet=featureSelect()

    f=open(file_in,'r')
    fw=open(file_out,'w')
    for line in f.readlines():
        lineSet=line.strip().split()
        string='1'+"\t"
        for i in range(0,len(featureSet)):
            if featureSet[i] in lineSet:
                string+=str(i+1)+":1 "
        print>>fw,string

#处理NB输出的结果
def nb_process():
    fr=open('nb0.out','r')
    fw=open('nb_tool','w')
    for line in fr.readlines():
        if line.strip()=='2':
            print>>fw,'N'
        else:
            print>>fw,'Y'




#合并情绪判断与情绪识别结果,也就是预测的结果
def merge_result(fileName1,fileName2,fileOut):
    dic={'1':'happiness','2':'like','3':'anger',
    '4':'sadness','5':'fear','6':'disgust','7':'surprise'}
    #svm_out
    f1=open(fileName1,'r')
    #result_emotion
    f2=open(fileName2,'r')
    f_out=open(fileOut,'w')
    #记录情绪识别结果
    emotion_label=[]
    for line in f1.readlines():
        emotion_label.append(dic[line.strip()])

    index=0
    for line in f2.readlines():
        if line.strip()=='N':
            temp='none'
            print>>f_out,temp
        else:
            temp=emotion_label[index]
            index=index+1
            print>>f_out,temp


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


 


if __name__=='__main__': 
   
#     featureSelect()
# #    #基于朴素贝叶斯进行有无情绪分类,多项式模型
    # result=NB('test_quzao_e.txt')
    #伯努利模型
    # result=NB1('test_quzao_e.txt')
 
#     f_temp=open('result_emotion_mu.txt','w')
#     for string in result:
#         print>>f_temp,string

    #朴素贝叶斯结果
#    result=[]
#    f=open('result_emotion_mu.txt','r')
#    for line in f.readlines():
#        result.append(line.strip())
    # result=[]
    # f=open('result_emotion_bo.txt','r')
    # for line in f.readlines():
    #     result.append(line.strip())

#    score(result)
    # train_for_nb("train_quzao_e","train_nb")
    # test_for_nb("test_quzao_e.txt","test_nb")
    # nb_process()
#    #SVM进行情绪识别
    # dic_intention_polarity=dic()
#    train_for_libsvm("weibo_train_emotion.txt_fenci","train_libsvm",dic_intention_polarity,0.9,0.1)
#    test_for_libsvm("weibo_test.txt_fenci",result,"test_libsvm",dic_intention_polarity,0.9,0.1)
     merge_result("NB_SVM_RESULT1","result_emotion_bo.txt","test.out5")
     score_emotion('label.txt','test.out5')  