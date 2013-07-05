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

perm=(1,3,0,2)
print perm
iperm=inv_P(perm)
print iperm
