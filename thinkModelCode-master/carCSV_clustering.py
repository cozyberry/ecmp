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
import string
from time import localtime, strftime, time

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
    print "%s [-c type_of_likelihood] [-n nonstochastic_iteration_times] [-s stochastic_iteration_times] [-v] [-l] [-o] [-d]"%sys.argv[0]
    print "     [-c type_of_likelihood]: 0 for normal likelihood;1 for classification likelihood;2 for naive bayesian network. 0 By default"
    print "     [-n iteration_times]: set nonstochastic iteration times for EM method. Default is 20"
    print "     [-s stochastic_iteration_times]: set stochastic iteration times for EM method. Default is 1"
    print "     [-v]: set verbose mode. Print other detail infomation"
    print "     [-l]: set objective of maximization of log likelihood; by default maximiation of score. Need to analysize further"
    print "     [-o]: output predicted class label and original label as well for further analysis"
    print "     [-d]: output file name with time stamp, only valid together with -o option"


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
    for k in range(np.size(xdata,1)): # loop thru number of columns in xdata
     if k==0:
      xdata_ml = lbin.fit_transform(xdata[:,k]);
     else:
      xdata_ml = np.hstack((xdata_ml,lbin.fit_transform(xdata[:,k])))
    #print "target set"
    #print np.unique(ydata)
    #ydata_ml = lbin.fit_transform(ydata)
    return numrows,xdata_ml,ydata,xdata,data

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

def buildNB(xtrain,ytrain):

    # ------------------------------ Naive_Bayes Model Construction ------------------------------
    # ------------------------------  MultinomialNB & ComplementNB  ------------------------------
    mnb = naive_bayes.MultinomialNB();
    mnb.fit(xtrain,ytrain);
    return mnb

"""
ltype stands for the type of likelihood:
    0 is normal likelihood
    1 is classification likelihood
"""
def calcObj(mnb,xtrain,ltype=0,ytrain=None):
    if ltype == 0:
        jll = mnb._joint_log_likelihood(xtrain)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        log_prob = np.sum(log_prob_x,axis=0)
        return log_prob
    elif ltype == 1:
        if ytrain == None :
            print "For the classification likelihood, please provide class label infomation!"
            sys.exit(1)
        else:
            numy=np.size(ytrain,0)
            numrows=np.size(xtrain,0)
            if numy != numrows:
                print "OH the number of attributes sample and the class label sets are inconsistent!"
                sys.exit(1)
            maxClass = ytrain[np.argmax(ytrain)]
            jll=mnb._joint_log_likelihood(xtrain) 
            numc = np.size(jll,1)
            #if maxClass >=numc:
                #print "Oh I don't have info about this class"
                #sys.exit(1)
            log_prob = 0.0
            #print "numc: %d"%numc
            #print "size of jll: %d * %d"%(np.size(jll,0),np.size(jll,1))
            for i in range(0,numy):
                #print "%d,%d"%(i,ytrain[i])
                if ytrain[i] < numc:
                    log_prob+=jll[i,ytrain[i]]
                else:
                    log_prob+=0.0
            return log_prob
    else:
        print "Oh I don't know how to calculate this type of likelihood ", ltype
        sys.exit(1)
    
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

def validate1(mnb,xtrain,ydata,numc):
    ypredict=mnb.predict(xtrain)
    allperms=itertools.permutations(range(0,numc))
    ydata0=np.zeros_like(ydata)
    numy=np.size(ydata,0)
    maxscore = 0.0
    #bestperm= allperms[0]
    for oneperm in allperms:
        for i in range(0,numy):
            ydata0[i]=oneperm[ydata[i]]
        #print oneperm
        #for j in range(numy-10,numy-1):
            #print "ydata[%d]: %d"%(j,ydata[j])
            #print "ydata0[%d]: %d"%(j,ydata0[j])
        tmpscore=mnb.score(xtrain,ydata0)
        if tmpscore > maxscore:
            maxscore = tmpscore
            bestperm=oneperm

    return maxscore,bestperm

def EMNB_csv(xtrain,ydata,numc,numrows,iterSN,iterCN):
    if _VERBOSE:
        prefix="clusteringLOG"
        if _DATE:
            outputDate=strftime("%m%d%H%M%S",localtime())
            logname="%s_%d_s%d_n%d_%s.csv"%(prefix,LTYPE,iterSN,iterCN,outputDate)
        else:
            logname="%s_%d_s%d_n%d.csv"%(prefix,LTYPE,iterSN,iterCN)
        log=open(os.path.join(LOGDIR,logname),'w')    
        print "NO_Class,NO_ITERSN,NO_ITERCN,LL,DIFF_CPT,ACCURACY,YET_CUR_BEST_LL,YET_CUR_BEST_ACCURACY,Comments"
        print >>log,"NO_Class,NO_ITERSN,NO_ITERCN,LL,DIFF_CPT,ACCURACY,YET_CUR_BEST_LL,YET_CUR_BEST_ACCURACY,Comments"
    best_accu= 0.0
    bestlog_prob = -float('inf') 
    for j in range(0,ITERSN):
    #Initializing step of target
        ydataf= -1*np.ones_like(ydata);
        for k in range(0,numrows):
            #randint is inclusive in both end
            ydataf[k]=random.randint(0,numc-1)
        ytrain=ydataf
    #initial
        mnb=buildNB(xtrain,ytrain)
        old_sigma_yx=np.array(np.zeros((numrows,numc)),float)
        diff = 10000.0
        for i in range(0,iterCN):
        #E-step
            sigma_yx=mnb.predict_proba(xtrain)
            diff_sig=sigma_yx-old_sigma_yx
            diff=LA.norm(diff_sig)
            old_sigma_yx=sigma_yx
        #M-step
            q_y=np.sum(sigma_yx,axis=0)/numrows 
            mnb.class_log_prior_=np.log(q_y)
            #alpha is very import to smooth. or else in log when the proba is too small we got -inf
            #ncx = safe_sparse_dot(sigma_yx.T, xtrain)+mnb.alpha
            ncx = safe_sparse_dot(sigma_yx.T, xtrain)+mnb.alpha
            ncxsum=np.sum(ncx,axis=1)
            qxy=np.divide(ncx.T,ncxsum).T
            mnb.feature_log_prob_=np.log(qxy)
            if _VERBOSE:
                if i%10 ==0 and i!=(iterCN-1):
                    log_prob=calcObj(mnb,xtrain)
                    tmpscore,tmpperm=validate(mnb,xtrain,ydata,numc)
                    print "%d,%d,%d,%f,%f,%f,%f,%f,Still in CN Loop"%(numc,j+1,i+1,log_prob,diff,tmpscore,bestlog_prob,best_accu)
                    print >>log,"%d,%d,%d,%f,%f,%f,%f,%f,Still in CN Loop"%(numc,j+1,i+1,log_prob,diff,tmpscore,bestlog_prob,best_accu)


        final_log_prob = calcObj(mnb,xtrain)
        score,tmpperm=validate(mnb,xtrain,ydata,numc)
        if _MAXLOG:
            if final_log_prob > bestlog_prob:
                _noconflict = True
                if score < best_accu:
                    _noconflict = False
                if _VERBOSE:
                    if _noconflict:
                        print "%d,%d,%d,%f,%f,%f,%f,%f,Better LL and NO Conflict"%(numc,j+1,iterCN,final_log_prob,diff,score,bestlog_prob,best_accu)
                        print >>log,"%d,%d,%d,%f,%f,%f,%f,%f,Better LL and NO Conflict"%(numc,j+1,iterCN,final_log_prob,diff,score,bestlog_prob,best_accu)
                    else:
                        print "%d,%d,%d,%f,%f,%f,%f,%f,Better LL but Conflict"%(numc,j+1,iterCN,final_log_prob,diff,score,bestlog_prob,best_accu)
                        print >>log,"%d,%d,%d,%f,%f,%f,%f,%f,Better LL but Conflict"%(numc,j+1,iterCN,final_log_prob,diff,score,bestlog_prob,best_accu)
                bestMNB = mnb
                bestlog_prob = final_log_prob
                best_accu = score
                best_perm=tmpperm
                best_iter = j
        else:
            if score > best_accu:
                if _VERBOSE:
                        print "%d,%d,%d,%f,%f,%f,%f,%f,Better Score"%(numc,j+1,iterCN,final_log_prob,diff,score,bestlog_prob,best_accu)
                        print >>log,"%d,%d,%d,%f,%f,%f,%f,%f,Better Score"%(numc,j+1,iterCN,final_log_prob,diff,score,bestlog_prob,best_accu)
                bestMNB = mnb
                bestlog_prob = final_log_prob
                best_accu = score
                best_perm=tmpperm
                best_iter = j
                    

    print "Best one is at %dth iteration"%best_iter
    print "The corresponding score: ", best_accu
    print "The corresponding log_prob: ", bestlog_prob
    print >>log,"Best one is at %dth iteration"%best_iter
    print >>log,"The corresponding score: ", best_accu
    print >>log,"The corresponding log_prob: ", bestlog_prob
    log.close()
    return bestMNB,best_perm

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
                ytrain[j]=np.nonzero(rclass_j[0])[0][0]
                #if i==0 and ytrain[j]==4:
                    #print "ytrain[%d]: %d"%(j,ytrain[j])
                    #print yproba_j
        else:
        #E-step and C-step
            ytrain = mnb.predict(xtrain)
        curnumc=np.size(np.unique(ytrain),0)
        if curnumc == 1:
            if _VERBOSE:
                print "Only One class is predicted. STOP earlier at %dth iteration"%i
                print >>log,"Only One class is predicted. STOP earlier at %dth iteration"%i
            break
    #M-step
        mnb=buildNB(xtrain,ytrain)
        #diffytrain=ytrain-oldytrain
        #diff=LA.norm(diffytrain)
        #print diff
        #if diff < 5:
        #    break

        log_prob=calcObj(mnb,xtrain,1,ytrain)
        if _VERBOSE:
            if i%10==0:
                tmpscore,tmpperm=validate(mnb,xtrain,ydata,numc)
                print "%d,%d,%s,%f,%f,%f"%(numc,i,i<iterSN,log_prob,log_prob-oldlog_prob,tmpscore)
                print >>log,"%d,%d,%s,%f,%f,%f"%(numc,i,i<iterSN,log_prob,log_prob-oldlog_prob,tmpscore)
        #print "%dth iteration gap of log_prob: %.15f"%(i,log_prob-oldlog_prob)
        if abs(log_prob - oldlog_prob) < stopGAP:
            if _VERBOSE:
                print "%f" %(log_prob-oldlog_prob)
                print "Converged. STOP earlier at %dth iteration"%i
                print >>log,"Converged. STOP earlier at %dth iteration"%i
            break
        oldlog_prob = log_prob

    score,perm=validate(mnb,xtrain,ydata,numc)
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
    testResult(testdata,xtest,ypredict,ytest,numc,np.size(xtest,0))

def NB_all(data,xdata_ml,ydata,numrows):
    #testdata,xtrain,ytrain,xtest,ytest=partition(numrows,data,xdata_ml,ydata)
    if _VERBOSE:
        print "Size of xtrain: %d * %d"%(np.size(xdata_ml,0),np.size(xdata_ml,1))

    mnb=buildNB(xdata_ml,ydata)
    print mnb.score(xdata_ml,ydata)
    numc=len(mnb.classes_)
    ypredict=mnb.predict(xdata_ml)
    testResult(data,xdata_ml,ypredict,ydata,numc,np.size(xdata_ml,0))

def main_v1(argv):
    try:
        opts, args = getopt.getopt(argv,"hc:n:s:vlod",["help"])
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
    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            sys.exit(0)
        elif opt in ("-c"):
            LTYPE = int(arg)
            if LTYPE != 0 and LTYPE !=1 and LTYPE!=2:
                print "Oh I don't know this type of likelihood: %d"
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

    random.seed()
    numrows,xdata_ml,ydata,xdata,data=initData(DATAPATH)
    #No need for spliting training data and testing data
    #xtrain,ytrain,xtest,ytest=partition(numrows,xdata_ml,ydataf)
    xtrain=xdata_ml
    #Right now it is the basic EM + NB model. Here we don't introduct stochastic operation
    if LTYPE ==0 or LTYPE ==1:
        print "nonstochastic iteration time is set at: " ,ITERCN
        print "stochastic iteration time is set at: " ,ITERSN
    
    if LTYPE == 0:
        numc=4
        mnb,perm=EMNB_csv(xtrain,ydata,numc,numrows,ITERSN,ITERCN)
    elif LTYPE == 1:
        numc=4
        mnb,perm=ECMNB(xtrain,ydata,numc,numrows,ITERSN,ITERCN)
    elif LTYPE == 2:
        NB_all(data,xtrain,ydata,numrows)

    if LTYPE ==0 or LTYPE ==1:
        ypredict0=mnb.predict(xtrain)
        ypredict=np.zeros_like(ypredict0)
        numy=np.size(ydata,0)
        for i in range(0,numy):
            ypredict[i]=perm[ypredict0[i]]
        testResult(data,xtrain,ypredict,ydata,numc,numrows)
    

def testResult(data,xdata,ypredict,ydata,numc,numrows):
    recall=np.array(np.zeros(numc),float)
    precision=np.array(np.zeros(numc),float)
    tp=np.array(np.zeros(numc),int)
    retrived=np.array(np.zeros(numc),int)
    relevant=np.array(np.zeros(numc),int)
    

    for i in range(0,numc):
        a=(ypredict==i)
        b=(ydata==i)
        tp[i]=np.sum(np.multiply(a,b))
        retrived[i]=np.sum(a)
        relevant[i]=np.sum(b)
        if relevant[i] != 0:
            recall[i]=np.float(tp[i])/relevant[i]
        else:
            recall[i] = 0.0
        if retrived[i] != 0:
            precision[i]=np.float(tp[i])/retrived[i]
        else:
            precision[i] = 0.0
        print "class %d: true_positive %d,retrived %d,relevant %d,recall %f, precision %f"%(i,tp[i],retrived[i],relevant[i],recall[i],precision[i])

    print "overall precision & recall: %f"%(np.float(np.sum(tp))/np.sum(retrived))
    #print "overall recall: %f"%(np.float(np.sum(tp))/np.sum(relevant))

    if _OUTPUT:
        outputDate=strftime("%m%d%H%M%S",localtime())
        prefix='carCluster'
        if _MAXLOG:
            prefix+='_l'
        if _DATE:
            outname="%s_%d_s%d_n%d_%s.csv"%(prefix,LTYPE,ITERSN,ITERCN,outputDate)
            outname_hu="%s_%d_s%d_n%d_%s_hu.csv"%(prefix,LTYPE,ITERSN,ITERCN,outputDate)
        else:
            outname="%s_%d_s%d_n%d.csv"%(prefix,LTYPE,ITERSN,ITERCN)
            outname_hu="%s_%d_s%d_n%d_hu.csv"%(prefix,LTYPE,ITERSN,ITERCN)

        title = ""
        for attr in ATTRIBUTES:
            title +="%s,"%attr
        #title = string.rstrip(title,',')
        title_hu=title
        title+='predicted_class,numerical_class,is_right'
        title_hu+='class,predicted_class,numerical_class,is_right'

        out=open(os.path.join(OUTPUTDIR,outname),'w')
        out_hu=open(os.path.join(OUTPUTDIR,outname_hu),'w')
        print >> out,title
        print >> out_hu,title_hu
        # To modify
        for i in range(0,numrows):
            onerow=""
            onerow_hu=""
            for item in xdata[i]:
                onerow+="%d,"%item
            for item in data[i]:
                onerow_hu+="%s,"%item
            onerow+="%d,%d,%s"%(ypredict[i],ydata[i],ypredict[i]==ydata[i])
            onerow_hu+="%d,%d,%s"%(ypredict[i],ydata[i],ypredict[i]==ydata[i])
            print >> out,onerow
            print >> out_hu,onerow_hu 

        out.close()

        print >>out_hu,""
        print >>out_hu,"statistique:"
        print >>out_hu,"class,true_positive,retrived,relevant,recall,precision"
        for i in range(0,numc):
            print >>out_hu,"%d,%d,%d,%d,%f,%f"%(i,tp[i],retrived[i],relevant[i],recall[i],precision[i])

        print >>out_hu,"overall precision & recall, %f"%(np.float(np.sum(tp))/np.sum(retrived))
        out_hu.close()

if __name__=='__main__':
    if len(sys.argv) > 1:
        main_v1(sys.argv[1:])
    else:
        main_v1("")
