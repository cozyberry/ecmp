#! /usr/bin/python
#A script to test functions. 

from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes
import numpy as np
import csv
import random
import urllib
import random
from numpy import linalg as LA
from sklearn.utils.extmath import safe_sparse_dot, logsumexp
import os,sys
import getopt
import itertools
import string
from time import localtime, strftime, time
from carCSV_clustering import inv_P
DATAPATH="/home/wei/data_processing/data/car/car.data"
ATTRIBUTES = ['buyPrice','maintPrice','numDoors','numPersons','lugBoot','safety']

def initData(filename):

    # Read in data from UCI Machine Learning Repository URL:
    #url = "http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    #webpage = urllib.urlopen(url)
    webpage=filename
    datareader = csv.reader(open(webpage,'r'))
    ct = 0;
    for row in datareader:
     ct = ct+1
    #webpage = urllib.urlopen(url) 
    datareader = csv.reader(open(webpage,'r'))
    data = np.array(-1*np.ones((ct,7),float),object);
    k=0;
    for row in datareader:
     data[k,:] = np.array(row)
     k = k+1;

    #To modify
    featnames = np.array(ATTRIBUTES,str)

    keys = [[]]*np.size(data,1)
    numdata = -1*np.ones_like(data);
    # convert string objects to integer values for modeling:
    for k in range(np.size(data,1)):
     keys[k],garbage,numdata[:,k] = np.unique(data[:,k],True,True)

    numrows = np.size(numdata,0); # number of instances in car data set
    numcols = np.size(numdata,1); # number of columns in car data set
    numdata = np.array(numdata,int)
    xdata = numdata[:,:-1]; # x-data is all data BUT the last column which are the class labels
    ydata = numdata[:,-1]; # y-data is set to class labels in the final column, signified by -1

    # ------------------ numdata multilabel -> binary conversion for NB-Model ---------------------
    lbin = LabelBinarizer();
    xclasses=[]
    for k in range(np.size(xdata,1)): # loop thru number of columns in xdata
     if k==0:
      #print "size of initial multi-value class 
      xdata_ml = lbin.fit_transform(xdata[:,k]);
      xclasses_= lbin.classes_
      print k
      print lbin.classes_
      xclasses.append(lbin.classes_)
     else:
      xdata_ml = np.hstack((xdata_ml,lbin.fit_transform(xdata[:,k])))
      xclasses_ = np.hstack((xclasses_,lbin.classes_))
      print k
      print lbin.classes_
      xclasses.append(lbin.classes_)
    #print "target set"
    #print np.unique(ydata)
    #ydata_ml = lbin.fit_transform(ydata)
      print "First two row of data:"
      print data[0:2,:]
      print "First two row of numdata:"
      print numdata[0:2,:]
      print "First two row of xdata:"
      print xdata[0:2,:]
      print "First two row of xdata_ml"
      print xdata_ml[0:2,:]
      print "classes"
      print xclasses_
      xclasses=xclasses_+1
      xdata_ml2=xdata_ml[0:2,:]
      tmp=np.multiply(xdata_ml2,xclasses_)
      print tmp
      tmp=tmp[np.nonzero(tmp)]
      #tmp=tmp[np.nonzero(tmp)].reshape(2,6)
      print tmp
      #xindices=np.nonzero(xdata_ml[0:2,:])
      #print "nonzero indices of first two row of xdata_ml"
      #print xindices


    return numrows,xdata_ml,ydata,xdata,data
if __name__=='__main__':
    initData(DATAPATH)

    #perm=(1,3,0,2)
    #print perm
    #iperm=inv_P(perm)
    #print iperm
