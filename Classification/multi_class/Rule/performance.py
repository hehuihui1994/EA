# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:05:41 2015

"""
from __future__ import division
import json,codecs,os
import math

def calc_acc(result,label):
    if len(result) != len(label):
        print 'Error: different lenghts!'
        return 0
    else:
        samelist = [int(str(x) == str(y)) for (x, y) in zip(result, label)]
#        samelist = [str(x) == str(y) for (x, y) in zip(result, label)]
        acc = float((samelist.count(1)))/len(samelist)
        return acc

def calc_recall(result,label,class_dict):
    '''class_dict中键为类别号，需要与label中保持一致；键值对应的是类别名称'''
    recall_dict = {}
    for l in class_dict:
        #判定为l且判定正确
        true_positive = sum([1 for (x, y) in zip(result,label) if x==l and y==l])
        #判定不是l但判断错误，且真实类别为l
        false_negative = sum([1 for (x, y) in zip(result,label) if x!=l and y==l])
        c = class_dict[l]
        if (true_positive+false_negative)!=0:
            recall_dict[c] = true_positive/(true_positive+false_negative)
        else:
            recall_dict[c] = 0
            
    return recall_dict
    

def calc_precision(result,label,class_dict):
    '''class_dict中键为类别号，需要与label中保持一致；键值对应的是类别名称'''
    precision_dict = {}
    for l in class_dict:
        #判定为l且判定正确
        true_positive = sum([1 for (x, y) in zip(result,label) if x==l and y==l])
        #print true_positive
        #判定为l但判断错误
        false_positive = sum([1 for (x, y) in zip(result,label) if x==l and y!=l])
        c = class_dict[l]
        if (true_positive+false_positive)!=0:
            precision_dict[c] = true_positive/(true_positive+false_positive)
        else:
            precision_dict[c] = 0
    return precision_dict

def calc_fscore(result,label,class_dict):
    '''计算F1值'''
    recall_dict = calc_recall(result,label,class_dict)
    precision_dict = calc_precision(result,label,class_dict)
    fscore_dict = {}
    for l in class_dict:
        c = class_dict[l] 
        fscore_dict[c] = fscore(recall_dict[c],precision_dict[c])
    return fscore_dict

def fscore(r,p):
    if r+p!=0:
        return (2*r*p)/(r+p)
    else:
        return 0


def calc_macro_average(result,label,class_dict):
    #计算宏平均recall,precision,F1
    recall_dict = calc_recall(result,label,class_dict)
    precision_dict = calc_precision(result,label,class_dict)
    class_num = len(class_dict.keys())
    macro_dict = {}
    macro_dict['macro_r'] = sum([recall_dict[class_dict[l]] for l in class_dict])/class_num
    macro_dict['macro_p'] = sum([precision_dict[class_dict[l]] for l in class_dict])/class_num
    macro_dict['macro_f1'] = fscore(macro_dict['macro_r'],macro_dict['macro_p'])
    
    return macro_dict

def calc_micro_average(result,label,class_dict):
    #计算微平均recall,precision,F1
    true_positive=0
    false_positive=0
    false_negative=0
    for l in class_dict:
        #判定为l且判定正确
        true_positive = true_positive + sum([1 for (x, y) in zip(result,label) if x==l and y==l])
        #判定为l但判断错误
        false_positive = false_positive +sum([1 for (x, y) in zip(result,label) if x==l and y!=l])
        #判定不是l但判断错误，且真实类别为l
        false_negative = false_negative + sum([1 for (x, y) in zip(result,label) if x!=l and y==l])


    micro_dict = {}
    micro_dict['micro_p']=true_positive/(true_positive+false_positive)
    micro_dict['micro_r']=true_positive/(true_positive+false_negative)
    micro_dict['micro_f1'] = fscore(micro_dict['micro_r'],micro_dict['micro_p'])
    
    return micro_dict


'''CCL2013计算宏平均recall,precision,F1'''
'''
def calc_macro_average(result,label,class_dict1):
    #计算宏平均recall,precision,F1
    recall_dict = calc_recall(result,label,class_dict1)
    precision_dict = calc_precision(result,label,class_dict1)
    #class_num = len(class_dict.keys())
    macro_dict = {}
    macro_dict['macro_r'] = sum([recall_dict[class_dict1[l]] for l in class_dict1])/6
    macro_dict['macro_p'] = sum([precision_dict[class_dict1[l]] for l in class_dict1])/6
    macro_dict['macro_f1'] = fscore(macro_dict['macro_r'],macro_dict['macro_p'])
    
    return macro_dict
'''

    
def calc_kappa(result,label,class_dict):
    '''计算kappa系数'''
    samp_num = len(result)
    po = sum([1 for (x, y) in zip(result,label) if x==y])/samp_num
    pe = 0
    for l in class_dict:
        pe += (result.count(l)*label.count(l))/(samp_num*samp_num)
    k = (po-pe)/(1-pe)
    return k

def demo_performance(result,label,class_dict):
    '''计算所有指标'''
    data = {}
    data['acc'] = calc_acc(result,label)
    recall = calc_recall(result,label,class_dict)
    precision = calc_precision(result,label,class_dict)
    fscore = calc_fscore(result,label,class_dict) 
    macro_avg = calc_macro_average(result,label,class_dict)
#    micro_avg = calc_micro_average(result,label,class_dict)
    data['kappa'] = calc_kappa(result,label,class_dict)
    
    #此处可用更新字典操作代替
    data['macro_r'],data['macro_p'],data['macro_f1'] = macro_avg['macro_r'],\
    macro_avg['macro_p'],macro_avg['macro_f1']
    
#    data['micro_r'],data['micro_p'],data['micro_f1'] = micro_avg['micro_r'],\
#    micro_avg['micro_p'],micro_avg['micro_f1']
    
    for l in class_dict:
        c = class_dict[l]
        data['r_'+c] = recall[c]
        data['p_'+c] = precision[c]
        data['f1_'+c] = fscore[c]
    return data
#    return acc,recall,precision,fscore, macro_average, micro_average,kappa

def demo_cv_performance(output_dir,fold_num,class_dict):
    '''计算n折交叉验证下的平均指标'''
    lst = []
    for fold_id in range(1,fold_num+1):
        fold_data = {}
        label_fname = output_dir+os.sep+'fold'+str(fold_id)+'\\ml\\test.samp'
        result_fname = output_dir+os.sep+'fold'+str(fold_id)+'\\ml\\mixed_test.result'
        label = [x.split('\t')[0] for x in open(label_fname).readlines()]
        result = [x.split('\t')[0] for x in open(result_fname).readlines()]
        fold_data['acc'] = calc_acc(result,label)
        fold_data['recall'] = calc_recall(result,label,class_dict)
        fold_data['precision'] = calc_precision(result,label,class_dict)
        fold_data['fscore'] = calc_fscore(result,label,class_dict)
        fold_data['macro_avg'] = calc_macro_average(result,label,class_dict)
#        fold_data['micro_avg'] = calc_micro_average(result,label,class_dict)
        fold_data['kappa'] = calc_kappa(result,label,class_dict)
        lst.append(fold_data)
    
    avg_data = {}
    avg_data['acc'] = sum([data['acc'] for data in lst])/fold_num
    avg_data['kappa'] = sum([data['kappa'] for data in lst])/fold_num
    avg_data['macro_r'] =  sum([data['macro_avg']['macro_r'] for data in lst])/fold_num
    avg_data['macro_p'] =  sum([data['macro_avg']['macro_p'] for data in lst])/fold_num
    avg_data['macro_f1'] =  sum([data['macro_avg']['macro_f1'] for data in lst])/fold_num
#    avg_data['micro_r'] =  sum([data['micro_avg']['micro_r'] for data in lst])/fold_num
#    avg_data['micro_p'] =  sum([data['micro_avg']['micro_p'] for data in lst])/fold_num
#    avg_data['micro_f1'] =  sum([data['micro_avg']['micro_f1'] for data in lst])/fold_num
    for l in class_dict.keys():
        c = class_dict[l]
        avg_data['r_'+c] = sum([data['recall'][c] for data in lst])/fold_num
        avg_data['p_'+c] = sum([data['precision'][c] for data in lst])/fold_num
        avg_data['f1_'+c] = sum([data['fscore'][c] for data in lst])/fold_num
    
    return avg_data
        

#def cv_avg_perform(input_dir,fold_num,class_dict):
#    lst = []
#    for fold_id in range(1,fold_num+1):
#        fold_data = {}
#        json_dir = input_dir+os.sep+'fold'+str(fold_id)+os.sep+'ml'+os.sep+'result.json'
#        #加载json数据
#        f = codecs.open(json_dir,'r','utf8')
#        chart_dict = json.load(f)
#        
#        result = [val['result'] for sq,val in chart_dict.iteritems()]
#        label = [val['label'] for sq,val in chart_dict.iteritems()]     
#        
#        
#
#        lst.append(fold_data)
#
#
#    
#    avg_json_dir = input_dir+os.sep+'avg_result.json'
#    f = codecs.open(avg_json_dir,'w','utf-8')  
#    json.dump(avg_data,f)
#    f.close()
        
    
    

def classify(score_fname,res_fname,pos_num,neg_num):
    score_list = [float(x.strip()) for x in open(score_fname).readlines()]
    res_list = []
    for i in range(len(score_list)):
        if score_list[i] > pos_num:
            res_list.append('1')
        elif score_list[i] < neg_num:
            res_list.append('-1')
        else:
            res_list.append('0')
    f = open(res_fname,'w')
    f.writelines([x+'\n' for x in res_list])
    f.close()
#    return res_list
    
if __name__ == '__main__':
    
#    acc_lst = []
#    for i in range(1,18):
#        output_dir = 'cross-domain1'+os.sep+'demo'+str(i)
#        
#        label = [x.strip() for x in open(output_dir+os.sep+'test'+os.sep+'test_label').readlines()]
#        result = [x.strip().split()[0] for x in open(output_dir+os.sep+'rule_mixed.result').readlines()]
#        
#        class_dict = {'1':'neg','2':'neu','3':'pos'}
#        result_dict = demo_performance(result,label,class_dict)
#        
#        print str(label.count('1'))+'\t'+str(label.count('2'))+'\t'+str(label.count('3'))+'\t',
#        ss = ''        
#        
#        for key in ['p_neg','r_neg','p_neu','r_neu','p_pos','r_pos','acc','kappa']:
#            ss += str(round(result_dict[key]*100,4))+'%\t'
#            if key=='kappa':
#                acc_lst.append(round(result_dict[key],4))
##            print str(round(result_dict[key]*100,4))+'%    ',
#        
#        print ss.rstrip('\t')
#    print sum(acc_lst)/len(acc_lst)
    
    output_dir = 'nlpcc'
    label = [x.strip() for x in open(output_dir+os.sep+'test'+os.sep+'test_label').readlines()]
    result = [x.strip().split()[0] for x in open(output_dir+os.sep+'mixed.result').readlines()]
    
    class_dict = {'1':'neg','2':'neu','3':'pos'}
    result_dict = demo_performance(result,label,class_dict)
    
    print len(label)
    print str(label.count('1'))+'\t'+str(label.count('2'))+'\t'+str(label.count('3'))
    
    ss = ''        
    
    for key in ['p_neg','r_neg','p_neu','r_neu','p_pos','r_pos','macro_f1','acc']:
        ss += str(round(result_dict[key]*100,4))+'%\t'
    print ss.rstrip('\t')
    
    
    
#    output_dir = 'nlpcc_objective'
#    label = [x.strip() for x in open(output_dir+os.sep+'test'+os.sep+'test_label').readlines()]
#    result = [x.strip().split()[0] for x in open(output_dir+os.sep+'svm.result').readlines()]
#    
#    class_dict = {'1':'por','2':'neu'}
#    result_dict = demo_performance(result,label,class_dict)
#    
#    print str(label.count('1'))+'\t'+str(label.count('2')),
#    ss = ''        
#    
#    for key in ['p_por','r_por','p_neu','r_neu','macro_p','macro_r','macro_f1','acc']:
#        ss += str(round(result_dict[key]*100,4))+'%\t'
#    print ss.rstrip('\t')
    
    
    
