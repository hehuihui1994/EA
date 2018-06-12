# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 16:09:33 2016
读入xml,提取weiboId,sentenceId,文字部分,其中weiboId,sentenceId并不影响情绪分类
@author: huihui
"""

import  xml.dom.minidom
import sys

reload(sys)
sys.setdefaultencoding('utf8')

def text():
    #打开xml文档
    dom = xml.dom.minidom.parse('test.xml')
    #得到文档元素对象
    root = dom.documentElement  
   #获得标签属性值
    itemlist = root.getElementsByTagName('weibo')

    f4=open('weibo1.txt','w')
    for item in itemlist:
        cc=item.getElementsByTagName('sentence')
        string=""
        for i in range(0,len(cc)):
            string+=cc[i].firstChild.data
        print>>f4,string
        
    
if __name__ == '__main__':
     text()