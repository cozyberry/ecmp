#! /usr/bin/python
from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes
import numpy as np
import csv
import random
import urllib
import random

random.seed()
rows=6
cols=7
data=np.ones((rows,cols),float)
nc=4
for i in range(0,rows):
    for j in range(0,cols):
        data[i,j]=random.randint(0,nc)
print "data size"
print "%d * %d"%(np.size(data,0),np.size(data,1))
sample=np.array(np.arange(rows),float)

print sample
sample2=2*sample
print sample2
res=np.multiply(sample,sample2)
print res 
