# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 15:49:33 2016
预处理
@author: huihui
"""

import re
import codecs

        
            
'''
去噪
'''
def remove_fileniose(filename,outName):
    #filename是文件夹
    infile=codecs.open(filename,'r',encoding='utf-8')
    outfile=codecs.open(outName,'w',encoding='utf-8')
    for line in infile.readlines():
        getline=remove_strnoise(line)
        outfile.write(getline)
        outfile.write('\n')

def remove_strnoise(content):
    #去除url
    pattern_url=re.compile(u'''
    (>>)*
    (((http|ftp|https)://)?
    (([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})
    |([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?)
    |((\d+)\.(\d+)\.(\d+)\.(\d+))
    ''',re.VERBOSE)
    replacedstr=re.sub(pattern_url,'',content)

    #去除用户id,形如：//@闲狂的蚂蚁:
    pattern_userid=re.compile(u'(//)@(\s)?([\u4e00-\u9fa5]|-|[a-zA-Z0-9]){1,20}(：|:)')
    replacedstr=re.sub(pattern_userid,'',replacedstr)
    # #将书名标记《》转换成【】
    # pattern_book=re.compile(u'(《|》)')
    # replacedstr=re.sub(pattern_book,'',replacedstr)

    #删除句末的@XXX
    pattern_lastid=re.compile(u'@([\u4e00-\u9fa5]|-|[a-zA-Z0-9]){1,15}(?=@|$|\s|】|\)|，|）|“)')
    replacedstr=re.sub(pattern_lastid,'',replacedstr)

    #去掉中间部分的@XXXX，无明显分隔符
    pattern_id=re.compile(u'@([a-zA-Z]|[\u4e00-\u9fa5]|-){1,10}')
    replacedstr=re.sub(pattern_id,'',replacedstr)

    #去除网站来源信息
    webfile=codecs.open('websites.txt','r',encoding='utf-8')
    websites=[]
    for site in webfile.readlines():
        websites.append(site.strip())
    for word in websites:
        replacedstr=replacedstr.replace(word,'')

    #去除时间
    pattern_time=re.compile(u'\d{4}-\d{,2}-\d{,2} \d{,2}:\d{,2}')
    if replacedstr:
        replacedstr=re.sub(pattern_time,'',replacedstr)

    #去除所有双斜跨//
    pattern_slash=re.compile(u'(/|\(|\)|）|（|\\\|\s|\.|？|\?)')
    if replacedstr:
        replacedstr=re.sub(pattern_slash,'',replacedstr)
    return replacedstr
      
    
if __name__ == '__main__':
#    remove_fileniose('weibo.txt','weibo_quzao.txt')    