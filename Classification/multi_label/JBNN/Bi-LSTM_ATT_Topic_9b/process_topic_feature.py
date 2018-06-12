#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 2017

@author: hehuihui1994 (he__huihui@163.com)
https://github.com/hehuihui1994
"""

def process_topic_file(n_topic):
    fr = open('k'+str(n_topic)+'.pz_d', 'r')
    fw = open('topic'+str(n_topic)+'.txt','w')
    num = 0
    for line in fr.readlines():
        if "nan" in line:
            print >> fw,"0."
            num+=1
        else:
            print>>fw,line.strip()
    print num
            


if __name__ == '__main__':
    process_topic_file(100)
    


