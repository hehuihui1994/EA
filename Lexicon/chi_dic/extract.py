# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 16:09:33 2016
读入xml
@author: huihui
"""

import  xml.dom.minidom
import sys

reload(sys)
sys.setdefaultencoding('utf8')
        

def emotion_label():
     #打开xml文档
    dom = xml.dom.minidom.parse('train.xml')    
    #得到文档元素对象
    root = dom.documentElement  
   #获得标签属性值
    itemlist = root.getElementsByTagName('weibo')
    #print itemlist[1].getAttribute("emotion-type")
    f=open('train_emotion_label.txt','w')
    for item in itemlist:
        if item.getAttribute("emotion-type")!="none" :
            print>>f,item.getAttribute("emotion-type")
            
   
def text1():
    #打开xml文档
    dom = xml.dom.minidom.parse('train.xml')
    
    #得到文档元素对象
    root = dom.documentElement  
   #获得标签属性值
    itemlist0 = root.getElementsByTagName('weibo')
    #print  cc[0].firstChild.data

    #输出文件
    f4=open('train_emotion.txt','w')
    f5=open('train_none_emotion.txt','w')     
     
    for item in itemlist0:
        string=""
        if item.getAttribute("emotion-type")!="none" :
            #获得标签对之间的数据
            cc=item.getElementsByTagName('sentence')
            for i in range(0,len(cc)):
                string=string+cc[i].firstChild.data
            print>>f4,string
        else:
            cc=item.getElementsByTagName('sentence')
            for i in range(0,len(cc)):
                string=string+cc[i].firstChild.data
            print>>f5,string
            
     
    
if __name__ == '__main__':
    text1()