#! /usr/bin/python

import os,sys,getopt,subprocess
from datetime import datetime,timedelta
from time import strftime
import re
import pprint,operator
from os.path import isdir,isfile
from proDataProcessing import myrexists
import string
from time import time
import shlex
from mpStats import printfromDictfile


ECFIELDFILE = '/home/wei/data_explore/scripts/confs/ecFields.csv'

START = "\033[1m"
END   = "\33[0;0m"

def usage():
    print START+"Usage:"+END
    print "      %s [-i indexToSortbyStringValue] [-I indexToSortbyIntegerValue] [-e] [-n numToDisplay] [-s dayToShift] [-p] [-d] [-t] [-a] dictFiletoPrint "%sys.argv[0]
    print START+"Options:"+END
    print "      [-i indexToSortbyStringValue]: specify which column to sort by using string value, index starting from 1\n"
    print "      [-I indexToSortbyIntegerValue]: specify which column to sort by using integer value, index starting from 1\n"
    print "      [-e] print field number of enriched coupon file as well\n" 
    print "      [-n numToDisplay]: specify number of lines to print to screen\n"
    print "      [-s dayToShift]: output last column as the date shifted"
    print "      [-p]: simply print fily by sorting"
    print "      [-d]: order from smallest to largest"
    print "      [-t]: input file with title"
    print "      [-a]: allign output according to field. Only works in print option"

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:I:en:s:pdta",["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    index = 0
    _ecfield = False
    isInt = False
    num = -1
    shift = 0 
    _print =False 
    _reverse = True
    _title=False
    _align=False

    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            sys.exit(0)
        elif opt in ("-i"):
            index=int(arg)-1
            if index < 0:
                print "Invalid index %s. Attention index starting from 1"%arg
        elif opt in ("-I"):
            index=int(arg)-1
            if index < 0:
                print "Invalid index %s. Attention index starting from 1"%arg
            isInt = True
        elif opt in ("-e"):
            _ecfield = True
        elif opt in ("-n"):
            num=int(arg)
        elif opt in ("-s"):
            shift=int(arg)
        elif opt in ("-p"):
            _print = True
        elif opt in ("-d"):
            _reverse = False
        elif opt in ("-t"):
            _title= True 
        elif opt in ("-a"):
            _align= True 
        else:
            usage()
            sys.exit(1)
    statinput = args[0]    

    if not os.path.exists(statinput):
        logMsg = "Error: The input stat file %s does not exist!"%statinput
        print logMsg
        sys.exit(2)

    f=open(statinput,'r')
    if _align:
        fLen=[]
    records=[]
    title=""
    if(_title):
        title=f.readline()
        if _align:
            titlefs=string.split(string.strip(title," ,\r\n"),",")
            for ti in range(0,len(titlefs)):
                    fLen.append(len(titlefs[ti]))
    for line in f:
        fields=string.split(string.strip(line,",\r\n "),",")
        if _align:
            for ti in range(0,len(fields)):
                    if len(fields[ti]) > fLen[ti]:
                        fLen[ti]=len(fields[ti])
                
        if isInt:
            fields[index]=int(fields[index])
        records.append(fields)

    sortedList = sorted(records,key=operator.itemgetter(index),reverse=_reverse)

    if num== -1 or num > len(sortedList):
        num=len(sortedList)
    if _title:
        if not _align:
            print title,
        else:
            print "here"
            print "len of titlefs[0]: %d"%len(titlefs[0])
            atitle=string.ljust(string.strip(titlefs[0]),fLen[0]," ")
            print "cur length of atitle: %d"%len(atitle)
            print "fLen[0]: %d"%fLen[0]
            for ti in range(1,len(titlefs)):
                atitle=atitle+","+string.ljust(titlefs[ti],fLen[ti]," ")
                print "cur length of atitle: %d"%len(atitle)
                print "fLen[%d]: %d"%(ti,fLen[ti])
            print atitle

    if _print:
        if not _align:
            for i in range(0,num):
                first = True
                tmpline=""
                for item in sortedList[i]:
                    if first:
                        tmpline=str(item)
                        first = False
                    else:
                        tmpline=tmpline+","+str(item)
                print tmpline
        else: 
            for i in range(0,num):
                item=sortedList[i][0]
                tmpline=string.ljust(str(item),fLen[0]," ")
                for itemj in range(1,len(sortedList[i])):
                    item=str(sortedList[i][itemj])
                    tmpline=tmpline+","+string.ljust(item,fLen[itemj]," ")
                print tmpline

    else:
        if _ecfield:
            fieldDict = dict()
            dictf=open(ECFIELDFILE,'r')
            for line in dictf:
                fields=string.split(string.strip(line),",")
                fieldDict[fields[0]]=int(fields[1])
            dictf.close()
        pattern=re.compile('\d{6,6}')

        for i in range(0,num):
            first = True
            tmpline=""
            for item in sortedList[i]:
                if first:
                    tmpline=item
                    first = False
                else:
                    tmpline=tmpline+","+str(item)

            mpdate=pattern.search(sortedList[i][0]).group() 
            if shift !=0:
                mpdate=datetime.strptime(mpdate,'%y%m%d')
                ecdate=mpdate+timedelta(days=shift)
                ecdate=ecdate.strftime("%y%m%d")
            else:
                ecdate=mpdate

            if not _ecfield:
                print tmpline+","+ecdate

            else:
                print tmpline+",%s,%d"%(ecdate,fieldDict[ecdate])
            


if __name__== '__main__':
    main(sys.argv[1:])

     
