#! /usr/bin/python

from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes
import numpy as np
from numpy import linalg as LA
from sklearn.utils.extmath import safe_sparse_dot, logsumexp
import os,sys
import string
from time import localtime, strftime, time
import copy
import itertools

class BaseMultinomialNBEM(naive_bayes.MultinomialNB):
    """
    a Multinomial Naive Bayes Cluster which combines [naive bayes classifier implementation] and [EM or ECM method] to deal with missing label information. 

    The Multinomial Naive Bayes Cluster is suitable for clustering with
    discrete features (e.g., word counts for text classification). 

    The Multinomial Naive Bayes Cluster can accept data without or with label information. 
    But the label information would only be used as informative guides

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    fit_prior : boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like, size=[n_classes,]
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    iterSN : iteration number for EM
    iterCN : retrial number for EM

    Attributes
    ----------

    n_cluster: initial maximum number of cluster

    local_prob_table: local probability table for each feature node. shape=[n_classes,n_features]

    featIndex: an internal array index for multi-value features. shape=[sum of domain length of each feature]

    nfeatures: an internal array of domain length of each multi-value feature shape=[feature number+1]

    `intercept_`, `class_log_prior_` : array, shape = [n_classes]
        Smoothed empirical log probability for each class.

    `feature_log_prob_`, `coef_` : array, shape = [n_classes, n_features]
        Empirical log probability of features
        given a class, P(x_i|y).

        (`intercept_` and `coef_` are properties
        referring to `class_log_prior_` and
        `feature_log_prob_`, respectively.)

    Examples
    --------
    To be modified
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, Y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2]))
    [3]

    Notes
    -----
    To be modified
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.
    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None,iterSN=10,iterCN=100):
        naive_bayes.MultinomialNB.__init__(self,alpha,fit_prior,class_prior)
        self._verbose = False
        self.outputdir = None 

    def setVerbose(self,verbose):
        self._verbose = verbose

    def setOutput(self,outputdir):
        self.outputDir = output 
        if (not os.path.exists(outputdir)) or (not os.path.isdir(outputdir)):
            raise ValueError("The output directory is invalide %s"%outputdir) 

        


    """
    if _labeled is True, the last column of data should be label information
    data is in the form of raw data, e.g. high,2,cheap,vgood
    """
    def transform_rawData(self,data,_labeled=True):

        keys = [[]]*np.size(data,1)
        numdata = -1*np.ones_like(data);
        nfeatures=[0]
        featIndex=[]

        # convert string objects to integer values for modeling:
        for k in range(np.size(data,1)):
         keys[k],garbage,numdata[:,k] = np.unique(xdata[:,k],True,True)

        numrows = np.size(numdata,0); # number of instances in car data set
        if _labeled:
            xdata = numdata[:,:-1]; # x-data is all data BUT the last column which are the class labels
            ydata = numdata[:,-1]; # y-data is set to class labels in the final column, signified by -1

        else:
            xdata = numdata
            ydata = None

        self.nfeatures=[0]
        lbin = LabelBinarizer();

        for k in range(np.size(xdata,1)): # loop thru number of columns in xdata
            if k==0:
                xdata_ml = lbin.fit_transform(xdata[:,k]);
                self.featIndex = lbin.classes_
                self.nfeatures.append(len(lbin.classes_))
            else:
                xdata_ml = np.hstack((xdata_ml,lbin.fit_transform(xdata[:,k])))
                self.featIndex= np.hstack((featIndex,lbin.classes_))
                self.nfeatures.append(nfeatures[-1]+len(lbin.classes_))

        return xdata_ml

    def inverse_transform(self,xdata_ml):
        numrows = np.size(xdata_ml,0)
        if(len(xdata_ml.shape) > 1):
            featIndex_t=np.tile(self.featIndex,(numrows,1))
            xdata_nz = (xdata_ml == 1)
            res = np.extract(xdata_nz,featIndex_t).reshape(numrows,-1)
            return res
        else:
            xdata_ml2=np.atleast_2d(xdata_ml)
            featIndex_t=np.atleast_2d(self.featIndex)
            return featIndex_t[:,np.nonzero(xdata_ml2)[1]]

    """
    ltype stands for the type of likelihood:
        0 is normal likelihood
        1 is classification likelihood
        2 is naive bayes with label information
    """
    def calcObj(xtrain,ltype=0,ytrain=None):
        if ltype == 0:
            jll = self._joint_log_likelihood(xtrain)
            # normalize by P(x) = P(f_1, ..., f_n)
            log_prob_x = logsumexp(jll, axis=1)
            log_prob = np.sum(log_prob_x,axis=0)
            return log_prob

        elif ltype == 1:
            if ytrain == None :
                raise ValueError("For the classification likelihood, please provide class label infomation!")
                sys.exit(1)
            else:
                numy=np.size(ytrain,0)
                numrows=np.size(xtrain,0)
                if numy != numrows:
                    raise ValueError("OH the number of attributes sample and the class label sets are inconsistent!")
                    sys.exit(1)

                maxClass = ytrain[np.argmax(ytrain)]
                jll=mnb._joint_log_likelihood(xtrain) 
                numc = np.size(jll,1)
                log_prob = 0.0

                for i in range(0,numy):
                    if ytrain[i] < numc:
                        log_prob+=jll[i,ytrain[i]]
                    else:
                        raise ValueError("Ah oh, something goes wrong")
                        log_prob+=0.0
                return log_prob
        else:
            raise ValueError("Oh I don't know how to calculate this type of likelihood %d "%ltype)
            sys.exit(1)

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
    def calCPT(self):
        jll = self.feature_log_prob_
        classArray = self.nfeatures

        nClass  =np.size(jll,0)
        nFeature=np.size(jll,1)

        ori_nFeature=len(classArray)-1
        if self._verbose:    
            print "nFeature: %d; nClass: %d; ori_nFeature: %d"%(nFeature,nClass,ori_nFeature)
        if ori_nFeature < 1 or nFeature != classArray[-1]:
            raise ValueError("the dimension of given jll: %d * %d is inconsistent with info of classArray: %d!"%(nFeature, nClass,classArray[-1]))
            return None

        nclassArray=classArray-np.roll(classArray,1)
        max_nClass=np.amax(nclassArray[1:])
        self.cpt=np.zeros((nClass,ori_nFeature,max_nClass))
        for i in range(0,nClass):
            for j in range(0,ori_nFeature):
                sumij=logsumexp(jll[i,classArray[j]:classArray[j+1]])
                for k in range(classArray[j],classArray[j+1]):
                    self.cpt[i,j,k-classArray[j]]=jll[i,k]-sumij

class MultinomialNBEM(BaseMultinomialNBEM):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None,iterSN=10,iterCN=100):
        naive_bayes.BaseMultinomialNBEM.__init__(self,alpha,fit_prior,class_prior,iterSN,iterCN)

    def build(self,n_cluster,data,_labeled=True,arrayAttr=None):
        if n_cluster <=1:
            raise ValueError("Please input a maximum cluster number no smaller than 1")
        if iterCN<=0:
            raise ValueError("Please input a strict positive integer value for iteration number of EM method")
        if iterSN<=0:
            raise ValueError("Please input a strict positive integer value for retrial number of EM method")

        self.n_cluster=n_cluster
        self.iterCN = iterCN
        self.iterSN = iterSN
        if arrayAttr != None:
            self.featnames = np.array(arrayAttr,str)
        self._labeled =_labeled
    """
    Transform multi-value xdata to binary representation of xdata_ml.
    For example, for a feature with 4 possible value: bad,good,vgood, the binary representation of a value 'bad' or 'good' or 'vgood' would be [1,0,0],[0,1,0] or [0,0,1] respectively
    """
        xdata_ml=transform_rawData(self,data,self._labeled)

        numrows = np.size(xdata_ml,0)
        numc = self.n_cluster
        if _verbose:
            prefix="NBEM"
            outputDate=strftime("%m%d%H%M%S",localtime())
            logname="%s_r%d_n%d_k%d_%s.csv"%(prefix,self.iterSN,self.iterCN,self.n_cluster,outputDate)
            log=open(os.path.join(self.outputDir,logname),'w')    

            print "NO_Class,NO_Trial,NO_ITER,LL,DIFF_CPT,YET_CUR_BEST_LL,Comments"
            print >>log,"NO_Class,NO_Trial,NO_ITER,LL,DIFF_CPT,YET_CUR_BEST_LL,Comments"
        bestlog_prob = -float('inf') 
        best_iter = 0
        best_class_prior = None 
        best_feature_log_prob = None
        for j in range(0,self.iterSN):
        #Initializing step of target
            ytrain = -1*np.ones(numrows);
            for k in range(0,numrows):
                #randint is inclusive in both end
                ytrain[k]=random.randint(0,numc-1)
        #initial
            naive_bayes.MultinomialNB.fit(self,xtrain,ytrain,class_prior=self.class_prior);

            old_sigma_yx=np.array(np.zeros((numrows,numc)),float)
            diff = 10000.0
            for i in range(0,iterCN):
            #E-step
                sigma_yx=naive_bayes.Multinomial.predict_proba(self,xtrain)
                diff_sig=sigma_yx-old_sigma_yx
                diff=LA.norm(diff_sig)
                old_sigma_yx=sigma_yx
            #M-step
                q_y=np.sum(sigma_yx,axis=0)/numrows 
                self.class_log_prior_=np.log(q_y)
                #alpha is very import to smooth. or else in log when the proba is too small we got -inf
                #ncx = safe_sparse_dot(sigma_yx.T, xtrain)+mnb.alpha
                ncx = safe_sparse_dot(sigma_yx.T, xtrain)+self.alpha
                ncxsum=np.sum(ncx,axis=1)
                qxy=np.divide(ncx.T,ncxsum).T
                self.feature_log_prob_=np.log(qxy)
# I am stopped here
                if self._verbose:
                    if i%10 ==0 or i > self.iterCN-5:
                        log_prob=calcObj(self,xtrain)
                        print "%d,%d,%d,%f,%f,%f,Still in CN Loop"%(numc,j+1,i+1,log_prob,diff,bestlog_prob)
                        print >>log,"%d,%d,%d,%f,%f,%f,Still in CN Loop"%(numc,j+1,i+1,log_prob,diff,bestlog_prob)


            final_log_prob = calcObj(self,xtrain)

            if final_log_prob > bestlog_prob:
                if self. _verbose:
                    print "%d,%d,%d,%f,%f,%f,Better LL and NO Conflict"%(numc,j+1,iterCN,final_log_prob,diff,bestlog_prob)
                        print >>log,"%d,%d,%d,%f,%f,%f,Better LL and NO Conflict"%(numc,j+1,iterCN,final_log_prob,diff,bestlog_prob)
                bestlog_prob = final_log_prob
                best_iter = j
                best_class_log_prior=copy.deepcopy(self.class_log_prior_)
                best_feature_log_prob=copy.deepcoay(self.feature_log_prob_)
            else:
                print "Ah oh I have no other criteria for choosing better model"
                        

        self.class_log_prior_=copy.deepcopy(best_class_log_prior)
        self.feature_log_prob_=copy.deepcoay(best_feature_log_prob)
        print "Best one is at %dth iteration"%best_iter
        print "The corresponding log_prob: ", bestlog_prob
        print >>log,"Best one is at %dth iteration"%best_iter
        print >>log,"The corresponding log_prob: ", bestlog_prob
        log.close()

