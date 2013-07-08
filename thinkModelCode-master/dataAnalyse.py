#! /usr/bin/python


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
ATTRIBUTES = ['buyPrice','maintPrice','numDoors','numPersons','lugBoot','safety']
OUTPUTDIR ='/home/wei/share/carClustering/analysis'

def usage():
    print "%s [-d] [-f filepath] [-o] [-n numrows]"%sys.argv[0]
    print "     [-d]: output original data as well"
    print "     [-o]: output to a file"
    print "     [-f filename]: input path of the file to be analyzed"
    print "     [-n numrows]: number of lines of the original file to be analyzed"

def main(argv=""):
    try:
        opts, args = getopt.getopt(argv,"hdof:n:",["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    printData = False 
    webpage = DATAPATH
    numrows = 0
    _output = False
    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            sys.exit(0)
        elif opt in ("-d"):
            printData = True
        elif opt in ("-f"):
            webpage=arg
        elif opt in ("-n"):
            numrows=int(arg)
        elif opt in ("-o"):
            _output = True

    datareader = csv.reader(open(webpage,'r'))
    ct = 0;
    for row in datareader:
        ct = ct+1
    datareader = csv.reader(open(webpage,'r'))
    data = np.array(-1*np.ones((ct,7),float),object);
    k=0;
    for row in datareader:
        data[k,:] = np.array(row)
        k = k+1;
    if numrows > 0 and numrows < ct:
        data=data[-numrows:]
    out = None 
    if _output:
        out=open(os.path.join(OUTPUTDIR,"a_"+os.path.splitext(os.path.basename(webpage))[0]+'.csv'),'w')
    initData(data,printData,out)
    if _output:
        out.close()


def initData(data,printData=False,out=None):
    if out==None:
        out=sys.stdout

    featnames = np.array(ATTRIBUTES,str)

    keys = [[]]*np.size(data,1)
    numdata = -1*np.ones_like(data);
    nkeys=[0]*np.size(data,1)
    # convert string objects to integer values for modeling:
    for k in range(np.size(data,1)):
     keys[k],garbage,numdata[:,k] = np.unique(data[:,k],True,True)
     nkeys[k]=len(keys[k])

    numrows = np.size(numdata,0); # number of instances in car data set
    numcols = np.size(numdata,1); # number of columns in car data set
    numdata = np.array(numdata,int)
    xdata = numdata[:,:-1]; # x-data is all data BUT the last column which are the class labels
    ydata = numdata[:,-1]; # y-data is set to class labels in the final column, signified by -1

    nFeature = np.size(xdata,1)
    numrows  = np.size(xdata,0)
    numc     = nkeys[-1]
    max_nFeature = np.amax(nkeys)
    fCount = np.zeros((numc,nFeature,max_nFeature))
    print >>out,"numc: %d; nFeature: %d; max_nFeature: %d"%(numc,nFeature,max_nFeature)  
    print >> out,"" 
    if printData:
        for i in range(0,numrows):
            line=data[i,0] 
            for j in range(1,nFeature+1):
                line+=",%s"%data[i,j]
            print >>out,line
        print >> out,"" 
    for i in range(0,numc):
        condition = (numdata[:,-1]==i).reshape(numrows,1)
        subnumrows= np.count_nonzero(condition)
        subindex  = np.nonzero(condition)
        #print "condition"
        #print condition
        condition = np.tile(condition,(1,nFeature))
        #print "condition"
        #print condition
        #print "xdata"
        #print xdata
        subMatrix = np.extract(condition,xdata).reshape(subnumrows,nFeature)
        #print "subMatrix"
        #print subMatrix
        for j in range(0,nFeature):
            submax    = np.amax(subMatrix[:,j])
            #print "nkeys[%d]: %d"%(j,nkeys[j])
            fCount [i][j][0:submax+1]=np.bincount(subMatrix[:,j])

    #print fCount

    for i in range(0,nFeature):
        title=ATTRIBUTES[i]
        for j in range(0,len(keys[-1])):
            title+=',%s'%keys[-1][j]
        print >>out,title
        for j in range(0,len(keys[i])):
            title=keys[i][j]
            for k in range(0,numc):
                title+=",%d"%fCount[k,i,j] 
            print >>out,title
        title = "total"
        for k in range(0,numc):
            title+=",%d"%np.sum(fCount[k,i,:])
        print >>out,title
        print >> out,"" 

            
    for i in range(0,nFeature):
        title=ATTRIBUTES[i]
        for j in range(0,len(keys[-1])):
            title+=',%s'%keys[-1][j]
        print >>out,title
        for j in range(0,len(keys[i])):
            title=keys[i][j]
            for k in range(0,numc):
                total = np.sum(fCount[k,i,:])
                frac = float(fCount[k,i,j])/total
                title+=",%f"%frac
            print >>out,title
        print >> out,"" 

    

if __name__=='__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main()
