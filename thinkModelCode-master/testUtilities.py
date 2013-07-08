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
    #xclasses=[]
    nclasses=[0]
    for k in range(np.size(xdata,1)): # loop thru number of columns in xdata
     if k==0:
      #print "size of initial multi-value class 
      xdata_ml = lbin.fit_transform(xdata[:,k]);
      xclasses_= lbin.classes_
      #print k
      #print lbin.classes_
      nclasses.append(len(lbin.classes_))
     else:
      xdata_ml = np.hstack((xdata_ml,lbin.fit_transform(xdata[:,k])))
      xclasses_ = np.hstack((xclasses_,lbin.classes_))
      nclasses.append(nclasses[-1]+len(lbin.classes_))
      #print k
      #print lbin.classes_
      #xclasses.append(lbin.classes_)
    #print "target set"
    #print np.unique(ydata)
    #ydata_ml = lbin.fit_transform(ydata)
    random.seed()
    rindex=(random.randint(0,numrows-1),random.randint(0,numrows-1))
    print "Random two row of data:"
    print data[rindex,:]
    print "Random two row of numdata:"
    print numdata[rindex,:]
    print "Random two row of xdata:"
    print xdata[rindex,:]
    print "Random two row of xdata_ml"
    print xdata_ml[rindex,:]
    print "classes"
    print xclasses_
    xdata_ml2=xdata_ml[rindex,:]
    xdata_2  =xdata[rindex,:]
    mnb=naive_bayes.MultinomialNB().fit(xdata_ml,ydata)
    print "mnb.feature_log_likelihood_"
    jll = mnb.feature_log_prob_
    print jll
    nclasses_1=np.roll(nclasses,1,axis=0)
    #print "ori_nclasses"
    #print nclasses
    print "nclasses"
    print (nclasses-nclasses_1)[1:]
    lct=calLCT(jll,nclasses)
    print "lct: %d * %d * %d"%(np.size(lct,0),np.size(lct,1),np.size(lct,2))
    explct=np.exp(lct)
    print explct 
    
    print "By naive_bayes funtion:"
    print mnb.predict_proba(xdata_ml2)
    print "By my funtion:"
    print predict_proba(xdata_2,lct,mnb.class_log_prior_)
    #print logsumexp(jll,axis=1)

    #xindices=np.nonzero(xdata_ml[0:2,:])
    #print "nonzero indices of first two row of xdata_ml"
    #print xindices
    return numrows,xdata_ml,ydata,xdata,data

"""
Parameters:
    jll: 
        type: numpy array; shape: [nclass_,nbinaryfeature_]
    classArray:
        type: list; format: for each row of jll, item indexes from classArray[i] to classArray[i+1](exclusive) is 
        the binary result of fiture i
Return:
    LCT table:
        type: ndarray; shape: [nclass_,n_oriFeature,max_nClass]
"""
def calLCT(jll,classArray):
    nClass  =np.size(jll,0)
    nFeature=np.size(jll,1)
    ori_nFeature=len(classArray)-1
    
    print "nFeature: %d; nClass: %d; ori_nFeature: %d"%(nFeature,nClass,ori_nFeature)
    if ori_nFeature < 1 or nFeature != classArray[-1]:
        print "the dimension of given jll: %d * %d is inconsistent with info of classArray: %d!"%(nFeature, nClass,classArray[-1])

    nclassArray=classArray-np.roll(classArray,1)
    max_nClass=np.amax(nclassArray[1:])
    lct=np.zeros((nClass,ori_nFeature,max_nClass))
    for i in range(0,nClass):
        for j in range(0,ori_nFeature):
            sumij=logsumexp(jll[i,classArray[j]:classArray[j+1]])
            for k in range(classArray[j],classArray[j+1]):
                lct[i,j,k-classArray[j]]=jll[i,k]-sumij
    return lct

def predict_proba(xdata,lct,class_log_prior):
    #step-1: proba(x|c)
    nClass  =np.size(lct,0)
    nFeature=np.size(lct,1)
    nSample =np.size(xdata,0)
    res=np.zeros((nSample,nClass))
    for i in range(0,nSample):
        for k in range(0,nClass):
            for j in range(0,nFeature):
                res[i,k]+=lct[k,j,xdata[i,j]]
    res=res+class_log_prior
    log_prob_x = logsumexp(res,axis=1)
    return np.exp(res-np.atleast_2d(log_prob_x).T)


if __name__=='__main__':
    initData(DATAPATH)

    #perm=(1,3,0,2)
    #print perm
    #iperm=inv_P(perm)
    #print iperm
