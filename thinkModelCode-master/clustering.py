#! /usr/bin/python
#carData_clustering.py

from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes
import numpy as np
import csv
import random
from numpy import linalg as LA
from sklearn.utils.extmath import safe_sparse_dot, logsumexp
import os,sys
import getopt
import itertools
import string
from time import localtime, strftime, time
import copy

DATAPATH="/home/wei/data_processing/data/car/car.data"
ITERCN = 20
ITERSN = 1
_VERBOSE = False
_MAXLOG  = False
_OUTPUT  = False
_DATE    = False
ATTRIBUTES = ['buyPrice','maintPrice','numDoors','numPersons','lugBoot','safety']
OUTPUTDIR ='/home/wei/share/carClustering/outputs/'
LOGDIR='/home/wei/share/carClustering/logs/'
LTYPE = 0

    
def usage():
    print "%s [-c type_of_likelihood] [-n nonstochastic_iteration_times] [-s stochastic_iteration_times] [-v] [-l] [-o] [-d] [-k initial clustering number]"%sys.argv[0]
    print "     [-c type_of_likelihood]: 0 for normal likelihood;1 for classification likelihood;2 for naive bayesian network. 0 By default"
    print "     [-n iteration_times]: set nonstochastic iteration times for EM method. Default is 20"
    print "     [-s stochastic_iteration_times]: set stochastic iteration times for EM method. Default is 1"
    print "     [-v]: set verbose mode. Print other detail infomation"
    print "     [-l]: set objective of maximization of log likelihood; by default maximiation of score. Need to analysize further"
    print "     [-o]: output predicted class label and original label as well for further analysis"
    print "     [-d]: output file name with time stamp, only valid together with -o option"
    print "     [-p]: set partition mode."
    print "     [-k initial clustering number]: set an initial clustering number for EMNB or ECMNB."


def initData(filename):
    if not os.path.exists(filename):
        print "I can't find this file: %s"%filename
        sys.exit(1)

    datareader = csv.reader(open(filename,'r'))
    ct = 0;
    for row in datareader:
     ct = ct+1

    datareader = csv.reader(open(filename,'r'))
    data = np.array(-1*np.ones((ct,7),float),object);
    k=0;

    for row in datareader:
     data[k,:] = np.array(row)
     k = k+1;

def partition1D(numrows,ydata):
    allIDX = np.arange(numrows);
    random.shuffle(allIDX); # randomly shuffles allIDX order for creating 'holdout' sample
    holdout_number = numrows/10; # holdout 10% of full sample set to perform validation
    testIDX = allIDX[0:holdout_number];
    trainIDX = allIDX[holdout_number:];
    ytest = ydata[testIDX];
    ytrain = ydata[trainIDX];
    return ytrain,ytest

def partition(numrows,data,xdata_ml,ydata):

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
    testdata=data[testIDX,:]
    return testdata,xtrain,ytrain,xtest,ytest


"""
Right now we only consider full mapping. man she
deprecated class 
"""
class classMap():
    def __init__(self,numc,curnumc):
        self.curIter = 0
        self.curnumc = curnumc
        self.numc    = numc

        if curnumc <= numc:
            self.nMap    = pow(curnumc,numc)
        else:
            print "Not implemented yet!"

    def next(self):
        while self.curIter < self.nMap:
            perm=[]
            a = self.curIter 
            for i in range(0,self.numc):
                a,b=divmod(a,self.curnumc)
                perm.append(b)
                
            self.curIter+=1
            if len(np.unique(perm)) == self.curnumc:
                return perm

        return None

    def allMaps(self,fromBeg=True):
        permSet=[] 
        if fromBeg:
            self.curIter = 0
        while True:
            perm = self.next()
            if perm != None:
                permSet.append(perm)
            else:
                break
        return permSet
            
    def printInfo(self):
        print "curIter: %d; curnumc: %d; numc:%d"%(self.curIter,self.curnumc,self.numc)


        

#########In clustering I don'ttttttttttttttttt care the classification label!!!!!!!!!!!!!!
#########Do not forgeeeeeeeeeeeeeeeeeeeeeeeeeet it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def ECMNB(xtrain,ydata,numc,numrows,iterSN,iterCN):
    if _VERBOSE:
        prefix="clusteringLOG"
        if _DATE:
            outputDate=strftime("%m%d%H%M%S",localtime())
            logname="%s_%d_s%d_n%d_%s.csv"%(prefix,LTYPE,iterSN,iterCN,outputDate)
        else:
            logname="%s_%d_s%d_n%d.csv"%(prefix,LTYPE,iterSN,iterCN)
        log=open(os.path.join(LOGDIR,logname),'w')    
        print "NO_Class,NO_ITER,is_S-step,CLL,DIFF_CLL,ACCURACY"
        print >>log,"NO_Class,NO_ITER,is_S-step,CLL,DIFF_CLL,ACCURACY"

    ydataf= -1*np.ones_like(ydata);
    for k in range(0,numrows):
        #randint is inclusive in both end
        ydataf[k]=random.randint(0,numc-1)
    ytrain=ydataf
#Initial step
    mnb=buildNB(xtrain,ytrain)
    iterTotal=iterSN+iterCN
    oldlog_prob=-float('inf')
    stopGAP = np.exp(-10)
#E-step and C-step or S-step
    for i in range(0,iterTotal):
        oldytrain=ytrain
        if i < iterSN:
            for j in range(0,numrows):
            #E-step
                yproba_j=mnb.predict_proba(xtrain[j])
            #S-step
                rclass_j=np.random.multinomial(1,yproba_j[0],size=1)
                #ytrain[j]=np.nonzero(rclass_j[0])[0][0]
                ytrain[j]=np.argmax(rclass_j[0])
                #if i==0 and ytrain[j]==4:
                    #print "ytrain[%d]: %d"%(j,ytrain[j])
                    #print yproba_j
        else:
        #E-step and C-step
            ytrain = mnb.predict(xtrain)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ykeys,ytrain = np.unique(ytrain,return_inverse=True)
        #curnumc=np.size(np.unique(ytrain),0)
        curnumc=len(ykeys)
        if i >= iterSN:
            print "%dth iteration curnumc: %d"%(i,curnumc)
            print "jll:"
            print oldmnb.coef_
        if curnumc == 1:
            if _VERBOSE:
                print "Only One class is predicted. STOP earlier at %dth iteration"%i
                print >>log,"Only One class is predicted. STOP earlier at %dth iteration"%i
            break
    #M-step
        oldmnb=copy.deepcopy(mnb)
        mnb=buildNB(xtrain,ytrain)
        #diffytrain=ytrain-oldytrain
        #diff=LA.norm(diffytrain)
        #print diff
        #if diff < 5:
        #    break
        
        log_prob=calcObj(mnb,xtrain,1,ytrain)
        if _VERBOSE:
            if i%20==0 or i >=iterSN:
                tmpscore,tmpperm=validate1(mnb,xtrain,ydata,numc)
                
                print "%d,%d,%s,%f,%f,%f"%(numc,i,i<iterSN,log_prob,log_prob-oldlog_prob,tmpscore)
                print >>log,"%d,%d,%s,%f,%f,%f"%(numc,i,i<iterSN,log_prob,log_prob-oldlog_prob,tmpscore)
                if tmpperm == None:
                    "Oh perm None"
                    break 
        #print "%dth iteration gap of log_prob: %.15f"%(i,log_prob-oldlog_prob)
        if log_prob - oldlog_prob < stopGAP and log_prob > oldlog_prob:
            if _VERBOSE:
                print "%f" %(log_prob-oldlog_prob)
                print "Converged. STOP earlier at %dth iteration"%i
                print >>log,"Converged. STOP earlier at %dth iteration"%i
            break
        oldlog_prob = log_prob
    print "mnb:"
    print mnb.coef_
    print mnb.intercept_
    mnb=oldmnb
    print "oldmnb jll:"
    print oldmnb.coef_
    print oldmnb.intercept_
    print "mnb:"
    print mnb.coef_
    print mnb.intercept_
    score,perm=validate1(mnb,xtrain,ydata,numc)
    log_prob=calcObj(mnb,xtrain,1,ytrain)
    #print "Best one is at %dth iteration"%best_iter
    print "The corresponding score: ",score 
    print "The corresponding log_prob: ", log_prob
    print >>log,"The corresponding score: ",score 
    print >>log,"The corresponding log_prob: ", log_prob
    log.close()
    return mnb,perm
 
def NB(data,xdata_ml,ydata,numrows):
    testdata,xtrain,ytrain,xtest,ytest=partition(numrows,data,xdata_ml,ydata)
    if _VERBOSE:
        print "Size of xtrain: %d * %d"%(np.size(xtrain,0),np.size(xtrain,1))

    mnb=buildNB(xtrain,ytrain)
    print mnb.score(xtest,ytest)
    numc=len(mnb.classes_)
    ypredict=mnb.predict(xtest)
    perm=tuple(range(0,numc))
    testResult(mnb,perm,testdata,xtest,ypredict,ytest,numc,np.size(xtest,0),nclasses)

#difference between NB_all and NB is just that NB_all use all data as trainning data as well as test data
def NB_all(data,xdata_ml,ydata,numrows):
    #testdata,xtrain,ytrain,xtest,ytest=partition(numrows,data,xdata_ml,ydata)
    if _VERBOSE:
        print "Size of xtrain: %d * %d"%(np.size(xdata_ml,0),np.size(xdata_ml,1))

    mnb=buildNB(xdata_ml,ydata)
    print mnb.score(xdata_ml,ydata)
    numc=len(mnb.classes_)
    ypredict=mnb.predict(xdata_ml)
    perm=tuple(range(0,numc))
    return mnb,perm

def main_v1(argv):
    try:
        opts, args = getopt.getopt(argv,"hc:n:s:k:vlodp",["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    global ITERCN
    global ITERSN
    global _VERBOSE
    global _MAXLOG
    global _OUTPUT
    global _DATE
    global LTYPE
    global OUTPUTDIR 
    global LOGDIR 
    _PARTITION = False
    numc = 4
    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            sys.exit(0)
        elif opt in ("-c"):
            LTYPE = int(arg)
            #if LTYPE != 0 and LTYPE !=1 and LTYPE!=2:
                #print "Oh I don't know this type of likelihood: %d"
        elif opt in ("-n"):
            ITERCN = int(arg)
        elif opt in ("-s"):
            ITERSN = int(arg)
        elif opt in ("-v"):
            _VERBOSE = True
        elif opt in ("-l"):
            _MAXLOG= True
        elif opt in ("-o"):
            _OUTPUT= True
        elif opt in ("-d"):
            _DATE= True
        elif opt in ("-p"):
            _PARTITION= True
        elif opt in ("-k"):
            numc = int(arg)

    random.seed()
    xdata_ml,xdata,ydata,data,nfeatures,keys,featIndex=initData(DATAPATH)
    numrows = np.size(xdata_ml,0)

    if _PARTITION:
        testdata,xtrain,ytrain,xtest,ytest=partition(numrows,data,xdata_ml,ydata)
    else:
        xtrain=xdata_ml
        ytrain=ydata
        testdata=data
        xtest=xtrain
        ytest=ydata
        
    #Right now it is the basic EM + NB model. Here we don't introduct stochastic operation
    if LTYPE ==0 or LTYPE ==1:
        print "nonstochastic iteration time is set at: " ,ITERCN
        print "stochastic iteration time is set at: " ,ITERSN
    
    if LTYPE == 0:
        mnb=EMNB_csv(xtrain,ytrain,numc,ITERSN,ITERCN)

    elif LTYPE == 1:
        mnb=ECMNB(xtrain,ytrain,numc,np.size(xtrain,0),ITERSN,ITERCN)

    elif LTYPE == 2:
        numc=len(keys[-1])
        mnb,perm=NB_all(data,xtrain,ydata,numrows)
        print "keys"
        print keys
        print "alpha: ",mnb.alpha

    testModel(mnb,testdata,xtest,ytest,nfeatures,keys,featIndex)
            

def testModel(mnb,data,xdata,ydata,nfeatures,keys,featIndex):
    numc = len(keys[-1])
    curnumc = len(mnb.classes_)
    numrows = np.size(xdata,0)
    ypredict = mnb.predict(xdata)

    dist = np.zeros((curnumc,numc))

    for i in range(0,curnumc):
        a=(ypredict==i)
        for j in range(0,numc):
            oj = (ydata == j)
            dist[i][j]=np.sum(np.multiply(a,oj))

    if _OUTPUT:
        outputDate=strftime("%m%d%H%M%S",localtime())
        prefix='nb_clustering'
        if _MAXLOG:
            prefix+='_l'
        if _DATE:
            outname="%s_%d_s%d_n%d_%s.csv"%(prefix,LTYPE,ITERSN,ITERCN,outputDate)
            outname_hu="%s_%d_s%d_n%d_%s_hu.csv"%(prefix,LTYPE,ITERSN,ITERCN,outputDate)
        else:
            outname="%s_%d_s%d_n%d.csv"%(prefix,LTYPE,ITERSN,ITERCN)
            outname_hu="%s_%d_s%d_n%d_hu.csv"%(prefix,LTYPE,ITERSN,ITERCN)

        out=open(os.path.join(OUTPUTDIR,outname),'w')
        out_hu=open(os.path.join(OUTPUTDIR,outname_hu),'w')

        title = ""
        for attr in ATTRIBUTES:
            title +="%s,"%attr
        title_hu=title

        title+='predicted_class,numerical_class'
        title_hu+='class,predicted_class,numerical_class'

        print >> out,title
        print >> out_hu,title_hu
        xdata_ori = inverse_transform(xdata,featIndex)
        for i in range(0,numrows):
            onerow=""
            onerow_hu=""
            for item in xdata_ori[i]:
                onerow+="%d,"%item
            for item in data[i]:
                onerow_hu+="%s,"%item
            onerow+="%d,%d"%(ypredict[i],ydata[i])
            onerow_hu+="%d,%d"%(ypredict[i],ydata[i])

            print >> out,onerow
            print >> out_hu,onerow_hu 

        out.close()
        lct=np.exp(calLCT(mnb.feature_log_prob_,nfeatures))
        printStats(dist,keys,lct)
        printStats(dist,keys,lct,out_hu)
        out_hu.close()

def printStats(dist,keys,lct,out=None):
    
    if out==None:
        out=sys.stdout

    print >>out, ""
    print >>out, "statistics of naive bayes model"
    print >>out, "number of class: %d; number of features: %d"%(np.size(lct,0),np.size(lct,1))
    for i in range (0,len(keys[-1])):
        print >>out, "%s ==> class %d"%(keys[-1][i],i)
    
    curnumc = np.size(lct,0)
    numc = len(keys[-1])
    _nfeat = len(keys)-1
    print >>out,""
    print >>out,"distribution:"
    title =""
    for i in range(0,curnumc):
        title+=',class %d'%i
    print >>out,title

    for i in range(0,numc):
        line=keys[-1][i]
        for j in range(0,curnumc):
            line+=",%d"%dist[j,i]
        print >>out,line

    print >>out,""
    print >>out,"characteristics:"
    outputLCT(lct,keys,out)


#Return an inverse of a permutation
def inv_P(perm):
    iperm=np.array(perm,int)
    for i in range(0,len(perm)):
        iperm[perm[i]] = i
    return iperm

def outputLCT(lct,keys,out=None):
    if out == None:
        out=sys.stdout
    nFeature = len(keys[-1])
    curnumc     = np.size(lct,0)

    for i in range(0,nFeature):
        title=ATTRIBUTES[i]
        for j in range(0,curnumc):
            title+=',class %d'%j
        print >>out,title
        for j in range(0,len(keys[i])):
            title=keys[i][j]
            for k in range(0,curnumc):
                frac = lct[k,i,j]
                title+=",%f"%frac
            print >>out,title
        print >> out,"" 

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
    if len(sys.argv) > 1:
        main_v1(sys.argv[1:])
    else:
        main_v1("")
