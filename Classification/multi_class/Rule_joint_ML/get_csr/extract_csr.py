# -*- coding: utf-8 -*-

#将连接词换成它们对应的类别序号
def get_con_int(file_bef,file_aft):
    file_path="conj"+"//"+"conjunction_words.txt"
    f=open(file_path,'r')
    dic_int_conj={}
    index=9
    for line in f.readlines():
        lineSet=line.strip().split()
        dic_int_conj[index]=lineSet
        index+=1
    # print dic_int_conj[0]
    fr=open(file_bef,'r')
    fw=open(file_aft,'w')
    for line in fr.readlines():
        lineSet=line.strip().split()
        string=""
        for word in lineSet:
            for key in dic_int_conj.keys():
                if word in dic_int_conj[key]:
                    word=str(key)
            string+=word+" "
        print>>fw,string

#输入sequence_input
def get_sequence_input():
    fw=open('sequence_input','w')
    train_y=[]
    fr_y=open("train_y_quchong_emotion",'r')
    for line in fr_y.readlines():
        train_y.append(line.strip())
    fr_x=open('train_x_quchong','r')
    index=0
    for line in fr_x.readlines():
        string=line.strip()+" @ "+train_y[index]
        index+=1
        print>>fw,string


#处理sub_train_x，去除只有连词的序列，去除连词连续的序列，去除train_x中已有的序列,最后一个是连词的也去除，去重
def get_sub_train_x_1_2():
    emotion=['happiness', 'like','anger','sadness','fear','disgust','surprise','none']
    fw1=open('sub_train_x_1','w')
    # fw2=open('sub_train_x_2','w')
    #train中已经有的序列
    train_csr_x=[]
    fr_train_x=open('train_x_quchong','r')
    for line in fr_train_x.readlines():
        lineSet=line.strip().split()
        train_csr_x.append(lineSet)
    #sub_train_x
    sub_train_x_quchong=[]
    fr_sub=open('sub_train_x','r')
    # fr_sub=open('sub_test.txt','r')
    for line in fr_sub.readlines():
        lineSet=line.strip().split()
        #1 检查是否只有连词
        flag=0
        for word in lineSet:
            if word in emotion:
                flag=1
                break
        if flag==0:
            continue

        #2 train_x中已有的序列
        if lineSet in train_csr_x:
            continue
        #3 连词连续的
        flag1=0
        if len(lineSet)>=2:
            for i in range(1,len(lineSet)):
                if (lineSet[i-1] not in emotion) and (lineSet[i] not in emotion):
                    flag1=1
                    break
        if flag1==1:
            continue
        #4最后一个是连词的去除
        if lineSet[len(lineSet)-1] not in emotion:
            continue
        #5 去重
        if lineSet not in sub_train_x_quchong:
            sub_train_x_quchong.append(lineSet)
            string=""
            for word in lineSet:
                string+=word+" "
            print>>fw1,string
    

#修改为class_supports
def get_fre_class():
    dic={}
    f=open('train_y_quchong_emotion','r')
    for line in f.readlines():
        temp=line.strip()
        if dic.has_key(temp):
            dic[temp]+=1
        else:
            dic[temp]=1
    for key in dic.keys():
        dic[key]=dic[key]*1.0/1289
        print("%r %r"%(key,dic[key]))
    return dic



##################CSR#####################
#找出sub_train_x_2中满足CSR规则的序列csr_sub
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

#一个集合的元素是否完全包含在另外一个集合中
def set_contain(set_son,set_father):
    for item in set_son:
        if item not in set_father:
            return 'false' 
    return 'true'

def satisfy_minsup_minconf(temp,csr_train_X,train_Y,fre):
    c=['happiness', 'like','anger','sadness','fear','disgust','surprise','none']
    #满足任何一个类别就return ture
    #总的CSR个数
    all_csr_num=len(train_Y)
    #找temp的父序列，存序列的下标,求出父亲序列个数以及父亲序列且类别相同的个数
    father_index=[]
    num_father=0
    for i in range(0,len(csr_train_X)):
        true_num=0
        j_index=0
        for temp_i in range(0,len(temp)):
            for j in range(j_index,len(csr_train_X[i])):
                if set_contain(temp[temp_i],csr_train_X[i][j])=='true':
                    true_num=true_num+1
                    j_index=j+1
                    break
        if true_num==len(temp):
            num_father=num_father+1
            father_index.append(i)
            # print i
    #num_father=0时，也就是不是任何规则的子序列
    if num_father==0:
        return 'false'
    #父序列且类别相同的个数
    #计算每个类别的support,confidence
    # print("num_father %r"%(num_father))
    # print("father_index %r"%(len(father_index)))

    num_class=0

    for c_class in c:
        num_class=0
        # print c_class
        # print("father_index %r"%(len(father_index)))
        for index in father_index:
            if train_Y[index]==c_class:
                # print("index train_Y[index] %r %r"%(index,train_Y[index]))
                num_class=num_class+1
        # print("num_class %r"%(num_class))
        support=num_class*1.0/all_csr_num
        confidence=num_class*1.0/num_father
        print("support:%r"%(support))
        print("confidence:%r"%(confidence))
        
        minconf=0.501
        minsup=0.04*fre[c_class]
        print("minsup:%r"%(minsup))
        print("minconf:%r"%(minconf))
        print("***************************")
        #同时满足，返回true
        if support >= minsup  and confidence >= minconf:
             # print c_class
             return c_class
    return 'false'




def csr_from_test():
    fw1=open('csr_sub_class_3','w')
    csr_x=[]

    #待分类序列
    csr_test_x=get_sequence_from_file('sub_train_x_2')
    

    #csr_train_X,train_Y读取
    csr_train_X=get_sequence_from_file('train_x_quchong_2')
    train_Y=[]
    f=open('train_y_quchong_emotion','r')
    for line in f.readlines():
       train_Y.append(line.strip())


    #去重后的CSR集合中的频率
    fre_dic=get_fre_class()
    
    for temp in csr_test_x:
        result_temp=satisfy_minsup_minconf(temp,csr_train_X,train_Y,fre_dic)
        if result_temp!='false':
            if temp in csr_x:
                continue
            else:
                csr_x.append(temp)
                #直接输出temp 
                string=""
                for item in temp:
                    if len(item)==1:
                        string=string+item[0]+" "
                    else:
                        string=string+item[0]+","+item[1]+" "
                string+=" @ "+result_temp
                print>>fw1,string
                print string
    return csr_x

#######################处理数据格式*******************************
#将训练集和测试集合处理成libsvm和NB都能接受形式，类别和特征之间是 \t
#x是temp的子序列
def contain_x(x,temp):
    j_index=0
    true_num=0
    for x_i in range(0,len(x)):
        for j in range(j_index,len(temp)):
            if set_contain(x[x_i],temp[j])=='true':
                true_num=true_num+1
                j_index=j+1
                break
    if true_num==len(x):
        return 'true'
    return 'false'

#只用CSR特征  csr_feature_set
#train_X2.txt   train_X2_for_tool
def train_to_libsvm_csr(fileName1,fileName2):
    f4=open(fileName2,'w')
    #读取csr特征
    csr=get_sequence_from_file('csr_feature_set_2')
    #处理数据变成LIBSVM接受的模式
    f2=open("train_label_int.txt",'r')
    train_label=[]
    for line in f2.readlines():
        line=line.strip()
        train_label.append(line)
    train_x=get_sequence_from_file(fileName1)
    for i in range(0,len(train_x)):
        string=train_label[i]+"\t"
        for j in range(0,len(csr)):
            if contain_x(csr[j],train_x[i])=='true':
            #contain改成=
            # if csr[j]==train_x[i]:
                string=string+str(j+1)+":1 "
        print>>f4,string

#test_X_dic2.txt test_X2_for_tool
def test_to_libsvm_csr(fileName1,fileName2):
    f4=open(fileName2,'w')
    #读取csr特征
    csr=get_sequence_from_file('csr_feature_set_2')
    #处理数据变成LIBSVM接受的模式
    train_x=get_sequence_from_file(fileName1)
    for i in range(0,len(train_x)):
        string="1"+"\t"
        for j in range(0,len(csr)):
            if contain_x(csr[j],train_x[i])=='true':
            # if csr[j]==train_x[i]:
                string=string+str(j+1)+":1 "
        print>>f4,string

#
def get_csr_sub0():
    f=open('csr_sub_class_3','r')
    fw=open('csr_feature_set','w')
    for line in f.readlines():
        lineSet=line.strip().split("@")
        lineSet=lineSet[0]
        print>>fw,lineSet




def get_csr_sub():
    f=open('csr_feature_set','r')
    fw=open('csr_feature_set_2','w')
    feature=[]
    for line in f.readlines():
        lineSet=line.strip().split()
        if lineSet in feature:
            continue
        else:
            feature.append(lineSet)
            string=""
            for word in lineSet:
                string+=word+" "
            print>>fw,string


if __name__ == '__main__':
    ####将测试集序列中的连词一体化######
    # file_path="test"+"//"
    # file1=file_path+"test_X_dic.txt"
    # file2=file_path+"test_X_dic2.txt"
    # get_con_int(file1,file2)
    ##########将训练集处理成train_x_quchong@train_y_quchong_emotion的格式，方便观察sequence_input#######
     # get_sequence_input()
     ##########处理sub_train_x##########
     # get_sub_train_x_1_2()
     #连词类型化
    # get_con_int('sub_train_x_1','sub_train_x_2')
     #########计算每个类别的频率############
     # get_fre_class()
     ##########从子序列中挖掘出满足条件的CSR,csr_sub_class_3######
     csr_from_test()
     ##########提取挖掘出的CSR中的X，'csr_feature_set#########
     # get_csr_sub0()
     #去重得到csr_feature_set_2
     # get_csr_sub()
     #处理成libsvm接受的形式
     # file_path_train="train"+"//"
     # file_path_train_bef=file_path_train+"train_X2.txt"
     # file_path_train_aft=file_path_train+"train_X2_for_tool"
     # train_to_libsvm_csr(file_path_train_bef,file_path_train_aft)
     # file_path_test="test"+"//"
     # file_path_test_bef=file_path_test+"test_X_dic2.txt"
     # file_path_test_aft=file_path_test+"test_X2_for_tool"
     # test_to_libsvm_csr(file_path_test_bef,file_path_test_aft)