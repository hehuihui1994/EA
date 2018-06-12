#!/usr/bin/env python
# encoding: utf-8


#评估在输出结果中排序第一的标签并不属于实际标签集中的概率
# outputs为预测的概率,test
def One_error(outputs,test):
    num_class = len(test[0])
    temp_outputs=[]
    temp_test=[]
    #1去除测试集真实标签全部为1或者全部为0的样例
    #这两种情况下没有排序第一
    for i in range(len(test)):
        sum_temp=0
        for j in range(len(test[i])):
            sum_temp += test[i][j]
        #所有类别都取到、所有类别都取不到
        if (sum_temp!=num_class)&(sum_temp!=0):
            temp_outputs.append(outputs[i])
            temp_test.append(test[i])

    #2接下来都用处理好的temp_outputs,temp_test
    num_class = len(temp_outputs[0])
    num_instance = len(temp_outputs)
    #测试集真实标签的类别集合
    label=[]
    for i in range(num_instance):
        temp_label=[]
        for j in range(len(temp_test[i])):
            if temp_test[i][j] == 1:
                temp_label.append(j)
        label.append(temp_label)

    #3
    OneError = 0
    for i in range(num_instance):
        indicator = 0
        #预测的情绪标签中的最大值以及其下标
        maximum = max(temp_outputs[i])
        for j in range(num_class):
            if temp_outputs[i][j] == maximum:
                if j in label[i]:
                    indicator=1
                    break
        #输出结果中排序第一的标签并不属于实际标签集
        if indicator==0:
            OneError+=1
    OneError=OneError*1.0/num_instance
    return OneError

from sklearn.metrics import label_ranking_loss, hamming_loss, f1_score, label_ranking_average_precision_score, accuracy_score, coverage_error
import numpy as np

def Emotion_eval(y_true, y_pred, y_pred_p):
    '''
    y_true = np.array([[1,0],[1,1]])
    y_pred = np.array([[1,1],[1,1]])
    y_pred_p = np.array([[0.9,0.6],[0.9,0.9]])
    '''

    #预测的标签
    print "HL : ",hamming_loss(y_true, y_pred)

    #预测的概率与真实标签
    print "RL : ",label_ranking_loss(y_true, y_pred_p)
    #预测的概率
    print "OE : ",One_error(y_pred_p, y_true)
    #预测的概率
    print "Coverage : ",coverage_error(y_true, y_pred_p)- 1.0
    # #标签
    # print "SA : ",accuracy_score(y_true, y_pred)
    # #标签
    # print "maF : ",f1_score(y_true, y_pred, average='macro')  
    # print "miF : ",f1_score(y_true, y_pred, average='micro')  
    #概率
    print "AP : ",label_ranking_average_precision_score(y_true, y_pred_p) 



''' 参考  https://github.com/hitalex/CCDM2014-contest/blob/master/calibrated_label_ranking.py'''

def calibrated_label_ranking(y_pred_p, For_calibrated_B):   
    y_pred = []
    for i in range(len(y_pred_p)):
        length = len(y_pred_p[i])
        temp = [0 for ii in range(length)]
        for j in range(length):
            for k in range(length):
                if j == k:
                    continue
                if y_pred_p[i][j] > y_pred_p[i][k]:
                    temp[j] += 1
            temp[j] += For_calibrated_B[i][j]

        y0 = (For_calibrated_B[i] < 1).sum()

        for i in range(length):
            if temp[i] > y0:
                temp[i] = 1
            else:
                temp[i] = 0
        y_pred.append(temp)
    return np.array(y_pred)



# if __name__ == '__main__':
#     y_pred_p = np.array([[0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0],[0, 0, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1]])
#     For_calibrated_B = np.array([[1, 1, 0, 0, 0, 0, 0, 0],[0, 0, 1, 1, 0, 0, 0, 0]])
#     y_pred = calibrated_label_ranking(y_pred_p , For_calibrated_B)
#     print y_pred