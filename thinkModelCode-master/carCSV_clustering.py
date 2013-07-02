#! /usr/bin/python
#carData_clustering.py

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

DATAPATH="/home/wei/data_processing/data/car/car.data"
def usage():
    print "%s [-n nonstochquqstic_iteration_times] [-s stochquastic_iteration_times] [-v]"
    print "     [-n iteration_times]: set nonstochquastic iteration times for EM method. Default is 20"
    print "     [-s stochquastic_iteration_times]: set stochquastic iteration times for EM method. Default is 10"
    print "     [-v]: set verbose mode. Print other detail infomation"


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
    #print "target set"
    #print np.unique(ydata)
    #ydata_ml = lbin.fit_transform(ydata)
    return numrows,xdata_ml,ydata

def partition1D(numrows,ydata):
    allIDX = np.arange(numrows);
    holdout_number = numrows/10; # holdout 10% of full sample set to perform validation
    testIDX = allIDX[0:holdout_number];
    trainIDX = allIDX[holdout_number:];
    ytest = ydata[testIDX];
    ytrain = ydata[trainIDX];
    return ytrain,ytest

def partition(numrows,xdata_ml,ydata):

    # -------------------------- Data Partitioning and Cross-Validation --------------------------
    # As suggested by the UCI machine learning repository, do a 2/3 train, 1/3 test split
    allIDX = np.arange(numrows);
    #random.shuffle(allIDX); # randomly shuffles allIDX order for creating 'holdout' sample
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

#def E_step(mnb,x):

def calcObj(mnb,xtrain):
    jll = mnb._joint_log_likelihood(xtrain)
    # normalize by P(x) = P(f_1, ..., f_n)
    log_prob_x = logsumexp(jll, axis=1)
    log_prob = np.sum(log_prob_x,axis=0)
    return log_prob
    
def validate(mnb,xtrain,ydata,numc):
    ypredict0=mnb.predict(xtrain)
    allperms=itertools.permutations(range(0,numc))
    ypredict=np.zeros_like(ypredict0)
    numy=np.size(ydata,0)
    maxscore = 0.0
    #bestperm= allperms[0]
    for oneperm in allperms:
        for i in range(0,numy):
            ypredict[i]=oneperm[ypredict0[i]]
        ydiff=ydata-ypredict
        csame=numy-np.count_nonzero(ydiff)
        tmpscore=float(csame)/numy
        if tmpscore > maxscore:
            maxscore = tmpscore
            bestperm=oneperm

    return maxscore,bestperm

    


def main_v1(argv):
    try:
        opts, args = getopt.getopt(argv,"hn:s:v",["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    iterCN = 20
    iterSN = 10
    _verbose = False
    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            sys.exit(0)
        elif opt in ("-n"):
            iterCN = int(arg)
        elif opt in ("-s"):
            iterSN = int(arg)
        elif opt in ("-v"):
            _verbose = True

    random.seed()
    numrows,xdata_ml,ydata=initData(DATAPATH)
    #No need for spliting training data and testing data
    #xtrain,ytrain,xtest,ytest=partition(numrows,xdata_ml,ydataf)
    xtrain=xdata_ml
    #Right now it is the basic EM + NB model. Here we don't introduct stochastic operation
    print "iteration time is set at: " ,iterCN
    
    numc=4
    for j in range(0,iterSN):
    #Initializing step of target
        ydataf= -1*np.ones_like(ydata);
        for i in range(0,numrows):
            #randint is inclusive in both end
            ydataf[i]=random.randint(0,numc-1)
        ytrain=ydataf

    #initial
        mnb=buildNB(xtrain,ytrain)
        if _verbose:
            print "number of class:", numc
        old_sigma_yx=np.array(np.zeros((numrows,numc)),float)

        for i in range(0,iterCN):
            #print i
    #E-step
            sigma_yx=mnb.predict_proba(xtrain)
            diff_sig=sigma_yx-old_sigma_yx
            diff=LA.norm(diff_sig)
            if _verbose:
                if i%10 ==0:
                    print "diff"
                    print "%d th iteration:%f"%(i,diff)
                    log_prob=calcObj(mnb,xtrain)
                    print "log_prob"
                    print "%d th iteration:%f"%(i,log_prob)

            old_sigma_yx=sigma_yx
    #S-step
            q_y=np.sum(sigma_yx,axis=0)/numrows 
            mnb.class_log_prior_=np.log(q_y)
        #alpha is very import to smooth. or else in log when the proba is too small we got -inf
            #ncx = safe_sparse_dot(sigma_yx.T, xtrain)+mnb.alpha
            ncx = safe_sparse_dot(sigma_yx.T, xtrain)+mnb.alpha
            ncxsum=np.sum(ncx,axis=1)
            qxy=np.divide(ncx.T,ncxsum).T
            mnb.feature_log_prob_=np.log(qxy)


        #ccount=np.array(np.bincount(ytrain),float)
        #ccount=ccount/numrows
        #t_m=np.multiply(ccount,ytrain)
        #score = mnb.score(xtrain,ydata)
        final_log_prob = calcObj(mnb,xtrain)
        score,tmpperm=validate(mnb,xtrain,ydata,numc)
        if _verbose:
            print "Classification accuracy of MNB = ", score
            print "Final Log Prob of MNB = ",final_log_prob
        if j==0:
            best_accu=score
            best_perm=tmpperm
            bestMNB = mnb
            bestlog_prob = final_log_prob
            best_iter = 0
        else:
            if final_log_prob > bestlog_prob:
                print "Get better"
                print "current best log prob vs this time: %f v.s. %f" %(bestlog_prob,final_log_prob)
                print "current best score vs this time: %f v.s %f" %(best_accu,score)
                bestMNB = mnb
                bestlog_prob = final_log_prob
                if score < best_accu:
                    print "Oh a conflict with score"
                print ""
                best_accu = score
                best_perm=tmpperm
                best_iter = j
    print "Best one is at %dth iteration"%best_iter
    print "The corresponding score: ", best_accu
    print "The corresponding log_prob: ", bestlog_prob

def main_v0():
    random.seed()
    numrows,xdata_ml,ydata=initData(DATAPATH)
#Initializing step of target
    ydataf= -1*np.ones_like(ydata);
    numc=3
    for i in range(0,numrows):
        ydataf[i]=random.randint(0,numc)
    #No need for spliting training data and testing data
    #xtrain,ytrain,xtest,ytest=partition(numrows,xdata_ml,ydataf)
    xtrain=xdata_ml
    ytrain=ydataf
    iterSN=10
    #iterCN=5
    iterCN=20
#E-step
    for i in range(0,iterSN):
        mnb=buildNB(xtrain,ytrain)
        #print i
        for j in range(0,numrows):
            yproba_j=mnb.predict_proba(xtrain[j])
            rclass_j=np.random.multinomial(1,yproba_j[0],size=1)
            ytrain[j]=np.nonzero(rclass_j[0])[0][0]
            #if i==0 and ytrain[j]==4:
                #print "ytrain[%d]: %d"%(j,ytrain[j])
                #print yproba_j
    for i in range(0,iterCN):
        #if i == 0:
            #print ytrain
        print i
        oldytrain=ytrain
        mnb=buildNB(xtrain,ytrain)
        ytrain=mnb.predict(xtrain)
        diffytrain=ytrain-oldytrain
        diff=LA.norm(diffytrain)
        print diff
        if diff < 5:
            break

    #ccount=np.array(np.bincount(ytrain),float)
    #ccount=ccount/numrows
    #t_m=np.multiply(ccount,ytrain)
    print "Classification accuracy of MNB = ", mnb.score(xtrain,ydata)

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

    print "One case:"
    print "    Attributes:"
    print xtest[1]
    print "    The predicted Proba:"
    print mnb.predict_proba(xtest[1])
    print "    The predicted Class:"
    print mnb.predict(xtest[1])

if __name__=='__main__':
    if len(sys.argv) > 1:
        main_v1(sys.argv[1:])
    else:
        main_v1("")
