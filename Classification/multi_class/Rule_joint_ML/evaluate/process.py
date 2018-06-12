# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:17:34 2016
计算各种指标
@author: huihui
"""
import performance

#读入人工标注的结果
def readin_label(filename):
    label=[]
    f=open(filename,'r')
    for line in f.readlines():
        labelLine=line.strip().split()
        label.append(labelLine[2])
    return label
    
#读入预测的标注
def readin_result(fileName):
    result=[]
    dic={'1':'happiness','2':'like','3':'anger',
    '4':'sadness','5':'fear','6':'disgust','7':'surprise',
    '8':'none'}
    f=open(fileName,'r')
    for line in f.readlines():
        index=line.strip()
        result.append(dic[index])
    return result

def merge(result1):
    result=[]
    f=open('result_emotion_tool.txt','r')
    index=0
    num=0
    for line in f.readlines():
        y_n=line.strip()
        if y_n=='Y':
            num+=1
            #修改
            # if result1[index]!='none':
            result.append(result1[index])        
        else:
            result.append('none')
        index+=1
#    print("merge %r"%(num))
    return result

#情绪判别
def get_result_sentiment(result):
    result_sentiment=[]
    num=0
    for word in result:
        if word=='none':
            # print word[0]
            result_sentiment.append('N')
        else:
            result_sentiment.append('Y')
            num+=1
#    print("get_result_sentiment %r"%(num))
    return result_sentiment



#情绪判断任务指标
def score(result):
    #人工标注
    label=[]
    f1=open('weibo_label.txt','r')
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


if __name__ == '__main__':
    #人工标注
    label=readin_label('weibo_label.txt')
    #预测出来的标注
    result=readin_result("result_haha6")
    result1=merge(result)
    
    result_sentiment=get_result_sentiment(result1)
    score(result_sentiment)
    
    #输出各类指标
    print("weibo情绪识别任务------------")        
    #人工标注
    label=readin_label('weibo_label.txt')
 
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
    
    
    
    
    
    
    
    
    
    