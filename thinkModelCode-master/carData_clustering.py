#! /usr/bin/python
#carData_clustering.py

from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes
import numpy as np
import csv
import random
import urllib
from sklearn.utils.extmath import safe_sparse_dot, logsumexp

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
    #testmnb0(mnb,xtest[0],True)
    print "Classification accuracy of MNB = ", mnb.score(xtest,ytest)

def predict_proba0(mnb,x):
    nc=np.size(mnb.intercept_,0)
    print "x:"
    print x
    ##calculate (1-2*x)*x_proba+x_proba
    #print "1-2*x:"
    #tmp=1-2*x
    #print tmp
    llp=mnb.feature_log_prob_
    print "llp:"
    print llp
    print "size of coef_: %d * %d"%(np.size(mnb.coef_,0),np.size(mnb.coef_,1))
    print mnb.coef_
    #res=np.multiply(tmp,llp)
    #print "res:"
    #print res
    sum1=np.sum(llp,axis=1)+mnb.intercept_
    print "sum1:"
    print sum1
    sumexp=np.exp(sum1)
    print sumexp
    sum2=np.sum(sumexp)
    print "after nomalized"
    norsum=sumexp/sum2
    print norsum
    print np.sum(norsum)
    
    #llp=np.inner(mnb.in
    
def testmnb0(mnb,x,allone=False):
    x0=np.array(np.zeros_like(x),int)
    print "scikit predict proba for all zero"
    print mnb.predict_proba(x0)
    if allone:
        x0=np.array(np.ones_like(x),int)

    print "scikit predict proba for all one"
    print mnb.predict_proba(x0)

    #predict_proba0(mnb,x0)
    

def testmnb(mnb,xtest,ytest):
    numrows=5
    xtest=xtest[0:numrows,:]
    print "size of xtest: %d * %d"%(np.size(xtest,0),np.size(xtest,1))
    print "%d samples:"%numrows
    print "Test Begins"
    jll = mnb._joint_log_likelihood(xtest)
    print "jll"
    print jll
    # normalize by P(x) = P(f_1, ..., f_n)
    log_prob_x = logsumexp(jll, axis=1)
    print "log_prob_x"
    print log_prob_x
    print "original intercept_"
    print mnb.intercept_

#E-step
    sigma_yx=mnb.predict_proba(xtest)
    print "sigma_yx: in fact the predict_proba"
    print sigma_yx
#S-step
    q_y=np.sum(sigma_yx,axis=0)/numrows 
    print "q_y"
    print q_y
    mnb.class_log_prior_=np.log(q_y)
    print "new intercep_:"
    print mnb.intercept_
    print "Original feature_log_prob_"
    print mnb.coef_
    print "Original sum test"
    print np.sum(np.exp(mnb.coef_),axis=1)
    #ncx = safe_sparse_dot(sigma_yx.T, xtest)
    ncx = safe_sparse_dot(sigma_yx.T, xtest)+mnb.alpha
    print "ncx"
    print ncx
    ncxsum=np.sum(ncx,axis=1)
    print "ncx sum operation"
    print ncxsum
    print "q_y"
    print q_y
    print "relation between q_y and ncxsum"
    print np.divide(ncxsum,q_y)

    qxy=np.divide(ncx.T,ncxsum).T
    print "test sum on qxy"
    print np.sum(qxy,axis=1)
    mnb.feature_log_prob_=np.log(qxy)
    print "new feature_log_prob_"
    print mnb.coef_
    print "new sum test"
    print np.sum(np.exp(mnb.coef_),axis=1)
    #print np.size(mnb.intercept_,0)
    print "feature_log_prob_"
    flp = mnb.feature_log_prob_
    print flp
    print "feature_log_prob_ size: %d * %d"%(np.size(flp,0),np.size(flp,1))
    print "xtest size: %d"%(np.size(xtest[0],0))
    #mnb.intercept_=np.array([[0 0 0 1]],float)
    #a=mnb.intercept_
    #mnb.intercept_=a
    print "test my llg"
    predict_proba0(mnb,xtest[0])

    print "mnb.intercept_ type"
    print type(mnb.intercept_)
    print "One case:"
    print "    Attributes:"
    print xtest[0]
    print "    The predicted Proba:"
    pvals=mnb.predict_proba(xtest[0])
    print "sum:"
    print np.sum(pvals)
    print pvals
    #rclass=np.random.multinomial(1,[1/6.]*6,size=1)
    rclass=np.random.multinomial(1,pvals[0],size=1)
    print "random generated class:"
    print rclass
    print "    The predicted Class:"
    print mnb.predict(xtest[0])
    print mnb.get_params(True)
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
