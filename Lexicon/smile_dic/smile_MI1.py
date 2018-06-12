# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math

#读取种子词
def read_seed():
    dic_seed={}
    seed_all=[]
    e=[[] for i in range(0,7)]
    emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise']
    f=open('seed11.txt','r')
    for line in f.readlines():
        lineSet=line.strip().split()
        seed_all.append(lineSet[0])
        for i in range(0,len(emotion)):
            if lineSet[1]==emotion[i]:
                e[i].append(lineSet[0])
    for i in range(0,len(emotion)):
        dic_seed[emotion[i]]=e[i]
    return dic_seed,seed_all

#每个表情出现的微博数,fileName为分好词的语料
def get_emotion_weibo_num(emoDic,fileName):
    dic_emotion_weibo_num={}
    for emotion in emoDic:
        dic_emotion_weibo_num[emotion]=0
    for i in range(0,len(emoDic)-1):
        for j in range(i+1,len(emoDic)):
            string1=emoDic[i]+emoDic[j]
            string2=emoDic[j]+emoDic[i]
            dic_emotion_weibo_num[string1]=0
            dic_emotion_weibo_num[string2]=0
    f=open(fileName,'r')
    #记录
    weibo_emotion_num=0
    index=0
    for line in f.readlines():
        lineSet=line.strip().split()
        text_emotion=[]
        for emotion in emoDic:
            if emotion in lineSet:
                dic_emotion_weibo_num[emotion]+=1
                text_emotion.append(emotion)
        #计数有表情符出现的微博数
        # if len(text_emotion)!=0:
        weibo_emotion_num+=1
            # index+=1
            # if index>20000:
            #     break
        #多个表情同时出现
        if len(text_emotion)!=1 and len(text_emotion)!=0:
            for i in range(0,len(text_emotion)-1):
                for j in range(i+1,len(text_emotion)):
                    string1=text_emotion[i]+text_emotion[j]
                    string2=text_emotion[j]+text_emotion[i]
                    dic_emotion_weibo_num[string1]+=1
                    dic_emotion_weibo_num[string2]+=1
    # print index
    return dic_emotion_weibo_num,weibo_emotion_num



#计算一个词与一个类别的点互信息
def MI(word,emotions_set,dic_emotion_weibo_num,weibo_emotion_num):
    #word与这个类别里面的所有的词都没有一起出现过,max_mi选默认的
    max_mi_default=-100000000
    mi=[]
    for smile in emotions_set:
        string=word+smile
        p_word_smile=dic_emotion_weibo_num[string]*1.0/weibo_emotion_num
#        print p_word_smile
        p_word=dic_emotion_weibo_num[word]*1.0/weibo_emotion_num
#        print p_word
        p_smile=dic_emotion_weibo_num[smile]*1.0/weibo_emotion_num
#        print p_smile
        #判断是否为0,若为0，这个smile没有参考意义
        if p_word_smile==0 or p_smile==0:
            continue
        temp=p_word_smile*1.0/(p_word*p_smile)
        result=math.log(temp)
        # if result<0.5:
        #     print word
        #     print smile
        #     print result
        mi.append(result)
    if len(mi)==0:
        max_mi=max_mi_default
        return max_mi
    #取出最大的Mi
    mi_temp=sorted(mi)
    mi_temp.reverse()
    max_mi=mi_temp[0]
    return max_mi
    


#读入待预测情绪，并且用互信息法进行预测
def predict_emotion(fileName):
    emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise']
    #种子
    dic_seed,seed_all=read_seed()
    #待预测词
    emoDic=[]
    f=open('EmoDic.txt','r')
    for line in f.readlines():
        lineSet=line.strip().split()
        emoDic.append(lineSet[0])
    dic_emotion_weibo_num,weibo_emotion_num=get_emotion_weibo_num(emoDic,fileName)
    #防止死循环
    now_len_emoDic=-1
    while(len(emoDic)!=0):
        pre_len_emoDic=len(emoDic)
        if pre_len_emoDic==now_len_emoDic:
            break
        for word in emoDic:
            #1:语料中没有这个表情的，无法判断，直接跳过,从列表中去除这个表情
            #2:dic_seed中已经有的不再计算,去除这个表情
            word_num=dic_emotion_weibo_num[word]
            if word_num==0 or (word in seed_all):
                emoDic.remove(word)
                continue
                
            #dic_seed中已经有的不再计算
            if word not in seed_all:
                mi=[0 for i in range(0,7)]
                for i in range(0,len(emotion)):
                    emotions_set=dic_seed[emotion[i]]
                    mi[i]=MI(word,emotions_set,dic_emotion_weibo_num,weibo_emotion_num)
                #z找出mi最大的那个所对应的类别
                mi_temp=sorted(mi)
                #最大值必须大于0.5 且 对应的情绪类别唯一
                num_max=0
                emotion_label=""
    
                if mi_temp[6]>0.5:
                    for i in range(0,len(emotion)):
                        if mi[i]==mi_temp[6]:
                            num_max+=1
                            emotion_label=emotion[i]
                #把word扩充到种子词中
                if num_max==1:
                    temp_set=dic_seed[emotion_label]
                    temp_set.append(word)
                    dic_seed[emotion_label]=temp_set
                    print word
                    print("emotion_label:%r mi_temp[6]:%r"%(emotion_label,mi_temp[6]))
                #3:=-100000000表示这个word没有与种子词里面的任何一个词一起出现过，还要继续判断
                #！=表示这个word已经被判断过了，种子词里面有跟这个word相关的词
                if mi_temp[6]!=-100000000:
                    emoDic.remove(word)
                now_len_emoDic=len(emoDic)
    #输出所有的已经有标签的表情
    f_w=open('mi_emo_dic2_test','w')
    for e in emotion:
        for smile in dic_seed[e]:
            string=smile+" "+e
            print string
            print>>f_w,string

# #读取服务器上的语料并且合并
# def merge():
#     #读取所有文件合并成weibo_all.txt
#     f_w=open('weibo_e.txt','w')
#     location="/home/poa/users/lyb/PUBLIC/yj/fenci_nlpir/"
#     date="2015-"
#     month=['05','06','07','08','09','10','11']
#     day_9=['00','01','02','03','04','05','06','07','08','09']
#     file_name=[]
#     for item in month:
#         for i in range(1,32):
#             if i<=9:
#                 string=date+item+"-"+day_9[i]
#                 # print string
#                 file_name.append(string)
#             else:
#                 if item=='06' or item=='09' or item=='11':
#                     if i!=31:
#                         string=date+item+"-"+str(i)
#                         # print string
#                         file_name.append(string)
#                 else:
#                     string=date+item+"-"+str(i)
#                     # print string
#                     file_name.append(string)
#     string='2015-12-01'
#     # print string
#     file_name.append(string)
#     for name in file_name:
#         string=location+name+"/"+name+".txt"
#         f=open(string,'r')
#         for line in f.readlines():
#             #去除空白
#             if len(line.strip())==0:
#                 continue
#             print>>f_w,line.strip()




if __name__ == '__main__':
   # predict_emotion('weibo_e.txt')
    # merge()
     predict_emotion('test_quzao_fenci_e.txt')