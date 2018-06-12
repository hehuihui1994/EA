# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 11:01:15 2016

@author: huihui
"""

import re

def count(fileName1,fileName2):
	f=open(fileName1,'r')
	dic_smile={}
	for line in f.readlines():
		smile=re.findall('\[.*?\]',line.strip())
		for item in smile:
			# print item
			if dic_smile.has_key(item):
				dic_smile[item]+=1
			else:
				dic_smile[item]=1
	dic_smile = sorted(dic_smile.iteritems(), key = lambda item:item[1], reverse=True )
	f_out=open(fileName2,'w')
	for word,value in dic_smile:
		string=word+" "+str(value)
		print >>f_out,string


if __name__ == '__main__':
	count('test_quzao_fenci_e.txt','test_quzao_fenci_e_smiles')