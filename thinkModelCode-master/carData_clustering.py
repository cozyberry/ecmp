#! /usr/bin/python
#carData_clustering.py

from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes
import numpy as np
import csv
import random
import urllib

DATAPATH="/home/wei/data_processing/data/car/car.data"

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

    featnames = np.array(['buyPrice','maintPrice','numDoors','numPersons','lugBoot','safety'],str)

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
    for k in range(np.size(xdata,1)): # loop thru number of columns in xdata
     if k==0:
      xdata_ml = lbin.fit_transform(xdata[:,k]);
     else:
      xdata_ml = np.hstack((xdata_ml,lbin.fit_transform(xdata[:,k])))
    print "target set"
    #print np.unique(ydata)
    #ydata_ml = lbin.fit_transform(ydata)
    return numrows,xdata_ml,ydata

def partition(numrows,xdata_ml,ydata):

    # -------------------------- Data Partitioning and Cross-Validation --------------------------
    # As suggested by the UCI machine learning repository, do a 2/3 train, 1/3 test split
    allIDX = np.arange(numrows);
    random.shuffle(allIDX); # randomly shuffles allIDX order for creating 'holdout' sample
    holdout_number = numrows/10; # holdout 10% of full sample set to perform validation
    testIDX = allIDX[0:holdout_number];
    trainIDX = allIDX[holdout_number:];

    # create training and test data sets
    xtest = xdata_ml[testIDX,:];
    xtrain = xdata_ml[trainIDX,:];
    ytest = ydata[testIDX];
    ytrain = ydata[trainIDX];
    return xtrain,ytrain,xtest,ytest

def buildNB(xtrain,ytrain):

    # ------------------------------ Naive_Bayes Model Construction ------------------------------
    # ------------------------------  MultinomialNB & ComplementNB  ------------------------------
    mnb = naive_bayes.MultinomialNB();
    mnb.fit(xtrain,ytrain);
    return mnb

def main():
    numrows,xdata_ml,ydata=initData(DATAPATH)
    xtrain,ytrain,xtest,ytest=partition(numrows,xdata_ml,ydata)
    mnb=buildNB(xtrain,ytrain)
    testmnb(mnb,xtest,ytest)
    print "Classification accuracy of MNB = ", mnb.score(xtest,ytest)

def testmnb(mnb,xtest,ytest):

    print "One case:"
    print "    Attributes:"
    print xtest[0]
    print "    The predicted Proba:"
    pvals=mnb.predict_proba(xtest[0])
    print pvals
    rclass=np.random.multinomial(1,pvals,size=1)
    print "random generated class:"
    print rclass
    print "    The predicted Class:"
    print mnb.predict(xtest[0])
'''
    print "One case:"
    print "    Attributes:"
    print xtest[1]
    print "    The predicted Proba:"
    print mnb.predict_proba(xtest[1])
    print "    The predicted Class:"
    print mnb.predict(xtest[1])
'''
if __name__=='__main__':
    main()