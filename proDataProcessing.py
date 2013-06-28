#! /usr/bin/python
import os,sys,getopt,subprocess
import pprint,operator
import string
from os.path import isdir,isfile
from time import localtime, strftime, time
import shlex
from subprocess import STDOUT
import re
try:
    from subprocess import DEVNULL # py3k
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

#[Variables setting section]
#shellconf: ecExtracted.conf <= the default extraction criteria configuration file
#indexconf: enrichedCouponIndex.txt <= the configuration file for cols to display
#awkfile: ecExtracted.awk <= the full file path of ecExtracted.awk. Please put this file in the same directory with this shell script
#outputDir: [shell script directory]/output <= the default directory for output files 
#defaut departure airport and arrival airport are RUH and JED
START = "\033[1m"
END   = "\33[0;0m"
SHELLPATH = os.path.dirname(os.path.abspath(sys.argv[0]))
CONFPATH = "%s/confs"                 %(SHELLPATH)
SHELLCONF= ""
INDEXCONF= ""
#SHELLCONF= "%s/ecExtracted.conf"       %(CONFPATH )
#INDEXCONF= "%s/enrichedcouponindex.txt"%(CONFPATH )
AWKFILE  = "%s/proDataProcessing.awk"       %(SHELLPATH)
#OUTPUTDIR= "%s/output"                %(SHELLPATH)
OUTPUTDIR="/home/wei/share/output"
OUTPUTFILE = ""
REMOTEPATH = "/remote/oridatacenter/PROCESSED/prorating_ond/"

INDEX64="%s/confs/enrichedcouponFullIndex64"%SHELLPATH
INDEX65="%s/confs/enrichedcouponFullIndex65"%SHELLPATH
INDEX71="%s/confs/enrichedcouponFullIndex71"%SHELLPATH
COLNUM=64
INDEXMAPFILE='/home/wei/data_explore/scripts/confs/mappingindex'
#[User defined fonction section]
def date2Filename(arg):
    monthDir ="20"+arg[0:2]+"-"+arg[2:4]
    filename ="enrichedcoupon"+arg+".csv"
    inputfile = os.path.join(REMOTEPATH,monthDir)
    inputfile =os.path.join(inputfile,filename)
    if not myrexists(inputfile):
        print START+"ERROR!"+END
        print '       The inputfile "%s" indicated by date:%s does not exist!'%(inputfile,arg)
        #usage()
        return "" 
        #sys.exit(2)
    return inputfile

def myrexists(path):
    try:
        cmd = "ls "+path
        res = subprocess.check_call(shlex.split(cmd),stdout=DEVNULL,stderr=STDOUT)
    except subprocess.CalledProcessError:
        return False
    return True

def readShellconf(_inputfile):
    params = dict()
    tmp = []
    f=open(_inputfile,'r')
    #oneSection = False
    for line in f:
        line = string.lstrip(line)
        if len(line) > 0:
            if line[0] == '#':
                continue
            elif line[0] == '[':
                #oneSection = True
                tmp = []
                tmp.append(string.strip(line[1:len(line)-1]))
            else:
                vals = string.split(line,'=')
                if len(vals) !=2:
                    print START+"ERROR!"+END
                    print '       Configuration file: %s format inrecognizable!'%_inputfile
                    usage()
                    sys.exit(2)
                tmp.append(string.strip(vals[1]))
                params[vals[0]]=tmp

    f.close()
    return params

def checkDir( dirTocheck, _mkdir = False ):
    if os.path.exists( dirTocheck ) and os.path.isdir( dirTocheck ):
        return True
    elif _mkdir == True:
        res = subprocess.call(["mkdir","%s"%dirTocheck])
        if res == 0:
            return True
    return False

    
def checkFile( fileTocheck, _newfile = False ):
    if os.path.exists( fileTocheck ) and os.path.isfile( fileTocheck ):
        return True
    elif _newfile == True:
        res = subprocess.call(["touch","%s"%fileTocheck])
        if res == 0:
            return True
    return False

#Help Message
def usage():
    print START+"Usage:"+END
    print "       %s [-h] [-c configuration_file_for_extraction] [-s configuration_file_for_output_columns] [-o outputdir] [-d date_of_prorated_tickets_file] [-e] inputfiles" % (sys.argv[0])
    #print "       %s [-h] [-c configuration_file_for_extraction] [-s configuration_file_for_output_columns] [-d] [-f] inputfiles_or_inputdates" % (sys.argv[0])
    print START+"Description:"+END
    print "       -h:"
    print "         Show usage"
    print "       -c configuration_file_for_extraction:"
    print "         Reading extraction criteria from \"configuration_file_for_extraction\"" 
    print "       -s configuration_file_for_output_columns:"
    print "         Output corresponding columns according to \"configuration_file_for_output_columns\""
    print "       -d:"
    print "         Instead of giving specific prarated tickets file path, it's possible to give just the date_of_prorated_tickets_file, eg. 130512 indicates the file: /remote/oridatacenter/PROCESSED/prorated_ond/2013-05/enrichedcoupon130512.csv."
    print "       -f:"
    print "         It's possible to guide the program to read input files or input dates from a file given by the argument."
    print "       -o outputdir:"
    print "         Print outputs to the specific directory \"outputdir\"" 
    print "       -e:"
    print "         Executation mode. No interactive actions" 

def main(argv):
    global OUTPUTFILE
#make sure that the conf file directory exists
    checkDir(CONFPATH,True)
#make sure that the output file directory exists
    checkDir(OUTPUTDIR,True)
#[Argument Processing section]
    try:
        opts, args = getopt.getopt(argv,"hc:s:dfo:e",["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    _shellconf = 0
    _indexconf = 0
    _output    = 0
    _date      = 0
    _filemode      = False
    _exe      =  False
#To be continued right now the confs are absolute locations input by user
    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            sys.exit()
        elif opt in ("-c"):
            _shellconf=1
            SHELLCONF=arg
            print "SHELLCONF is %s"%arg
            if not checkFile(SHELLCONF):
                print START+"ERROR!"+END
                print "       The configuration_file_for_extraction:%s does not exist!"%SHELLCONF
                usage()
                sys.exit(2)
        elif opt in ("-s"):
            _indexconf = 1
            INDEXCONF = arg
            if not checkFile(INDEXCONF):
                print START+"ERROR!"+END
                print "       The configuration_file_for_output_columns:%s does not exist!"%INDEXCONF
                usage()
                sys.exit(2)
        elif opt in ("-o"):
            _output = 1
            tmpoutdir=arg
        elif opt in ("-d"):
            _date = 1
        elif opt in ("-f"):
            _filemode= True
        elif opt in ("-e"):
            _exe = True

#[Executation log file setting section]
    logfile=open(SHELLPATH+"/ecExtracted_shelllog",'a')
    print >>logfile,""
    exDate=strftime("%Y-%m-%d %H:%M:%S", localtime())
    print >>logfile,"Last executation at "+exDate 
    #if (_date==0 and len(args) != 1) or (_date==1 and len(args) > 0):
    if len(args) == 0 :
        errMsg ="Too Few or Too Many Arguments" 
        print >> logfile,errMsg 
        print START+"ERROR!"+END
        print "       "+errMsg
        logfile.close()
        usage()
        sys.exit(2)
    outputDate=strftime("%m%d%H%M%S",localtime())

#Acquire the inputfile path, as convention it should be the only non-option argument
    #output="${outputDir}/ecEx${inputbase}"
    #if _date == 0:
        #inputfile = args[0]
    #print >>logfile,inputfile
    if not _filemode:
        if _date == 1:
            inputfiles=[]
            for datearg in args:
                inputfiles.append(date2Filename(datearg))
        else:
            inputfiles=args
    else:
        print "Datemode: %d"%_date
        inputfiles=[]
        for arg in args:
            if not os.path.exists(arg):
                print "Error! File:%s does not exist!"%arg
                sys.exit(2)
            f=open(arg,'r')
            for line in f:
                strs=string.split(string.strip(line),",")
                if _date == 1:
                    inputfiles.append(date2Filename(strs[0]))
                else:
                    inputfiles.append(strs[0])
            f.close()


#[Beginning Extraction section] 
    onds = ""
    dds  = ""
    cconf =""
    if _shellconf == 0:
        if _exe:
            print "In executation mode, configuration file needed!"
            sys.exit(2)
        while onds == "":
            print "Please enter:Origin and Destination Pair sequences, eg. LONMAD_MADLON"
            onds=raw_input()
        print "Please enter: Departure Date sequences, eg. 130522_130530"
        dds=raw_input()

        ondsStr = onds[0][0:3]+'-'+onds[0][3:6]
        i =1
        while i < len(onds):
            ondsStr = ondsStr+"/"+onds[i][0:3]+'-'+onds[i][3:6]
            i = i+1

        ddsStr = ""
        if dds == "":
            logMsg = " No constraints on departure date sequences." 
            print START+"INFO:"+END+logMsg
            print >>logfile,logMsg
        else:
            ddsStr=string.replace(dds,'_','/')


    #print "Starting searching items concerning  $dept  ->  $arri  depature during  $ddFrom  ~  $ddTo"


#[Reading Index of Columns to display if -s option is set]
    fieldDict = dict()
    dictf=open(INDEXMAPFILE,'r')
    for line in dictf:
        fields=string.split(string.strip(line),",")
        if len(fields)>0:
            fieldDict[fields[0]]=int(fields[len(fields)-1])
    dictf.close()

    ORISHELLCONF = SHELLCONF


#To be completed
    for inputfile in inputfiles:
        if not os.path.exists(inputfile):
            continue

        tmpfile=open(inputfile,'r')
        tmpline = tmpfile.readline()
        tmpcls  = string.split(string.strip(tmpline),",") 
        COLNUM = len(tmpcls)
        if len(tmpcls[len(tmpcls)-1]) == 0:
            COLNUM = COLNUM -1
        print COLNUM
        if _shellconf and COLNUM == 71:
            tmpshellconf = os.path.splitext(ORISHELLCONF)[0]+'71.csv'
            if not os.path.exists(tmpshellconf):
                tmpshellfile=open(tmpshellconf,'w')
                print tmpshellconf
                oriconffile=open(ORISHELLCONF,'r')
                for oriline in oriconffile:
                    oriline=string.strip(oriline)
                    if len(oriline) == 0:
                        continue
                    confpattern= re.compile("^[\t ]*#")
                    b=confpattern.search(oriline)
                    if confpattern.search(oriline) != None:
                        #print "Comment"
                        print>>tmpshellfile,oriline
                        print oriline
                    else:
                        #print "Not comment" 
                        oricols=string.split(oriline,",")
                        tmpline=str(fieldDict[oricols[0]])
                        for j in range(1,len(oricols)): 
                            tmpline=tmpline+",%s"%oricols[j]
                        print >>tmpshellfile,tmpline
                tmpshellfile.close()
                oriconffile.close()

            SHELLCONF=tmpshellconf
        else:
            SHELLCONF=ORISHELLCONF
            
        strIndex = ""
        if _indexconf== 1:
        #read needed enriched coupon INDEX columns from enrichedCouponIndex.txt
        #output needed index to tmpIndex
            indexFile = open(INDEXCONF,'r')
            strIndexname = ""
            for line in indexFile:
                tmp = string.split(line,',')    
                if COLNUM==64 or COLNUM==65 or COLNUM==62:
                    strIndex = strIndex+tmp[0]+" "
                    strIndexname = strIndexname+tmp[1]+" "
                elif COLNUM==71:
                    strIndex = strIndex+str(fieldDict[tmp[0]])+" "
                    strIndexname = strIndexname+tmp[1]+" "
            indexFile.close()

        if _output == 0:
            if SHELLCONF!="": 
                tmpoutdir = os.path.splitext(os.path.basename(ORISHELLCONF))[0]
                tmpoutdir = os.path.join(OUTPUTDIR,tmpoutdir)
                checkDir(tmpoutdir,True)
            else:
                tmpoutdir = OUTPUTDIR

        OUTPUTFILE = os.path.basename(inputfile)
        OUTPUTFILE = OUTPUTFILE[0:len(OUTPUTFILE)-4]+'.csv'
        OUTPUTFILE = os.path.join(tmpoutdir,OUTPUTFILE)

        awkcmd = []
        awkcmd.append('awk')
        awkcmd.append('-f')
        awkcmd.append('%s'%AWKFILE)
        if _shellconf == 0:
            awkcmd.append('-v')
            awkcmd.append('onds=%s'%ondsStr)
            awkcmd.append('-v')
            awkcmd.append('dds=%s'%ddsStr)
        else :
            awkcmd.append('-v')
            awkcmd.append('conf=%s'%SHELLCONF)
        awkcmd.append('-v')
        awkcmd.append('cols=%s'%strIndex)
        awkcmd.append('-v')
        awkcmd.append('out=%s'%OUTPUTFILE)
        awkcmd.append('%s'%inputfile)
        #print "py awk"
        print >>logfile,awkcmd
        #awkcmd1= 'awk -f /home/wei/data_explore/script_examples/ecExtracted.awk -v "onds=LON-MAD/MAD-LON" -v "dds= 130522/130530" -v "cols=" -v "out=" "/remote/oridatacenter/PROCESSED/prorating_ond/2013-05/enrichedcoupon130512.csv"'
        #awkcmd2=shlex.split(awkcmd1)
        #print awkcmd2
        logMsg = " Start extraction......" 
        print START+"INFO:"+END+logMsg
        print >>logfile,logMsg
        logfile.flush()
        start = time()
        retcode = subprocess.call(awkcmd)
        exectime = time() - start
        logMsg ="Finished. Used %f secondes for file: %s"%(exectime,inputfile)
        print START+"INFO: "+END+logMsg
        print >>logfile,logMsg
        print >> logfile, "the retcode is %d for file: %s"%(retcode,inputfile)
        if retcode > 0 and myrexists(OUTPUTFILE):
            outputOri=open(OUTPUTFILE, 'r')
            outputCopy = os.path.basename(inputfile)
            outputCopy = outputCopy[0:len(outputCopy)-4]+'withTitle.csv'
            outputCopydir = os.path.join(tmpoutdir,"withTitle")
            checkDir(outputCopydir,True)
            outputCopy = os.path.join(outputCopydir,outputCopy)
            outputCopy=open(outputCopy,'w')
            title = ""
            if _indexconf== 0:
                if retcode == 71:
                    rightIndex=open(INDEX71,'r')
                elif retcode == 64 or retcode ==65 or retcode ==62:
                    rightIndex=open(INDEX65,'r')
                else:
                    outputCopy.close()
                    outputOri.close()
                    logfile.close()
                    return retcode 
                i = 0
                while i < retcode:
                    i = i+1
                    line = string.strip(rightIndex.readline())
                    title = title+line.split(",")[1]+","
            else:
                rightIndex=open(INDEXCONF,'r')
                for line in rightIndex:
                    line = string.strip(line)
                    title = title+line.split(",")[1]+","

            rightIndex.close()

            firstLine = True
            for line in outputOri:
                if firstLine:
                    firstLine = False
                    firstline = string.strip(line)
                    if firstline[len(firstline)-1] != ',':
                        title=title[0:len(title)-1]
                    print >> outputCopy,title
                print >>outputCopy,line,
            outputCopy.close()
            outputOri.close()

    logfile.close()
    #return retcode



if __name__== '__main__':
    if len(sys.argv) == 1:
        print START+"ERROR!"+END
        print "       Missing arguments!"
        usage()
        sys.exit(2)
    pstart = time()
    main(sys.argv[1:])
    pexectime = time() - pstart
    logMsg ="Task Finished. Used %f secondes."%pexectime
    print START+"INFO: "+END+logMsg
##echo $inputfile
#awkcmd="awk -f $awkfile -v dept=$dept -v arri=$arri -v ddFrom=$ddFrom -v ddTo=$ddTo -v out=$output -v cols=\"$strIndex\" \"$inputfile\""
#echo $awkcmd>>"$logfile"
##`$awkcmd`
#awk -f $awkfile -v dept=$dept -v arri=$arri -v ddFrom=$ddFrom -v ddTo=$ddTo -v out=$output -v cols="$strIndex" $inputfile
    
