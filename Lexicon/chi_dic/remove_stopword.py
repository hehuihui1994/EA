# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:47:12 2016

@author: huihui
"""

import codecs

#去除停用词
def del_stopwords(infilename,outfilename):
    stopwordsfile=codecs.open('stopword.txt','r',encoding='utf-8')
    outfile=codecs.open(outfilename,'w',encoding='utf-8')
    stopwordslist=[]
    for line in stopwordsfile.readlines():
        stopwordslist.append(line.strip())
    stopwordsfile.close()
    infile=codecs.open(infilename,'r',encoding='utf-8')
    for line in infile.readlines():
        line=line.encode('gbk','ignore').decode('gbk')
        for word in line.strip().split():
            if word not in stopwordslist:
                outfile.write(word+' ')
        outfile.write('\n')
    outfile.close()
    infile.close()

if __name__ == '__main__':
    #del_stopwords("train_extract_quzao.txt_fenci","train_quzao_fenci_removeStopWord.txt")
     del_stopwords("train_emotion_quzao.txt_fenci","train_removeStopWord.txt")
