#! /usr/bin/python

""""Description for all the fonctions I wrote
updateStat(): 
"""
import os,sys,getopt,subprocess
import pprint,operator
from os.path import isdir,isfile
from proDataProcessing import myrexists
import string
from time import time
import shlex

START = "\033[1m"
END   = "\33[0;0m"
SCRIPTDIR = os.path.dirname(os.path.abspath(sys.argv[0]))
STATSDIR  = "%s/stats/"%SCRIPTDIR
STATSFILE = "%s/mpStats.dat"%STATSDIR
ECEXTRACTSCRIPT = "%s/proDataProcessing.sh"%SCRIPTDIR
MPDAILYDIR = "/remote/oridatacenter/RAW_DATA/mp_daily/"
MPDAILYPRODIR = "/remote/oridatacenter/PROCESSED/mp_daily/"
ECSUMMARY =  "/remote/oridatacenter/PROCESSED/prorating_ond/summary_stats/"
PRORATING_OND = "/remote/oridatacenter/PROCESSED/prorating_ond/"
PRODATASCRIPT = "proDataProcessing.py"
_EXE = False
_UPDATE=False

def printBold(msg):
    global START
    global END
    print START+msg+END


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


#Check initial environnement settings
def init():
    checkDir( STATSDIR, True)
    if not checkFile( STATSFILE ):
        updateStat( MPDAILYDIR )
    if not checkFile ( ECEXTRACTSCRIPT ):
        print START+"Error!"+END
        print "       The assciated shell script: \"%s\" does not exist!"%ECEXTRACTSCRIPT
        sys.exit(2)


def usage():
    global START
    global END

    print START+"Usage:"+END
    print "       %s [-u] [-s OriginDestination] [-a] [-n number_to_display] [-m origin_and_destination_city_pari_sequence[_departure sequence]] [-e] [-g .|origin_and_destination_city_pari_sequence[_departure sequence]] [directoryToLookIn]\n" % (sys.argv[0])
    print START+"Description:"+END
    print "       directoryToLookIn:"
    print "         The default path \"%s\" will be set when directoryToLookIn is not specified.\n"%MPDAILYDIR
    print "      -u:"
    print "         Update master pricer search statistics using up-to-date information. Used with -s -a or -n options\n"
    print "      -s OriginDestination:"
    print "         Specify to look at search statistics of a particular pair. OriginDestination is a string consisted by 6 letters whose first 3 letters are the city code of origin and whose latter 3 letters are that of destination, e.g. PARLON.\n"
    print "      -a:"
    print "         To display complete statistics of master pricer searches. This option is the default one when no options are specified\n"
    print "       -n number_to_display:"
    print "          Specify how many results to display, ordered by search frequency of master pricer. Used with -s or -a option.\n"
    print "       -m origin_and_destination_city_pari_sequence[_departure date sequence]"
    print "          Look at the summary of bookings with certain OnD sequence, e.g. LON-MAD or LON-MAD/MAD-LON or LON-MAD/MAD-LON_130530/130531. Used with -a or -n option. Not compatible with -s option\n"
    print "       -e:"
    print "         Executation mode. No interactive actions" 
    print "       -g .|origin_and_destination_city_pari_sequence[_departure sequence]"
    print "          Look at the summary of bookings with certain OnD sequence, e.g. LON-MAD or LON-MAD/MAD-LON or LON-MAD/MAD-LON_130530/130531. When using with . look at the whole distribution. Used with -a or -n option. Not compatible with -s option\n"
    print START+"Output:"+END
    print "       Statistics of master pricer searches and a csv format file of prorated tickets with respect to particular criteria\n"
    print START+"Attention:"+END
    print "       The options: [-s OriginDestination] or [-a] are mutually exclusive. And the option [-a] is the default one. And [-s OriginDestination] option is considered when [-a] is also specified" 



def mpSummaryDistribution(dirname,_ond,_dds,num=-1,out=""):
    if not os.path.isdir(dirname):
        print "Error! %s is not a valid directory path"%dirname
        sys.exit(2)
    files = os.listdir(dirname)
    sortedfiles = sorted(files)
    n = len(string.split(_ond,"/"))
    ondDict = dict()
    for onedir in sortedfiles:
        onedir=os.path.join(dirname,onedir)
        if os.path.isdir(onedir):
            _onds=string.split(string.upper(_ond),"/")
            _ddss=string.split(string.strip(_dds),"/")
            if len(_onds) != len(_ddss):
                print "Error in format:%s_%s The length should be the same"%(_ond,_dds)
            else:
                #pattern ="*"
                pattern =""
                i = 0
                while i<len(_onds):
                    pattern=pattern+string.replace(_onds[i],"-","")+_ddss[i]
                    i=i+1
                #pattern=pattern+"*"
                pattern=pattern
                allfiles = os.listdir(onedir)
                for onefile in allfiles:
                    if string.find(onefile,pattern) > 0:
                        dictKey=os.path.basename(onedir)
                        ondDict[dictKey]=onefile
                        break
    #            findcmd='find %s -type f -name "%s"'%(onedir,pattern)
                ##llcmd='ls -l  %s'%onedir
                ##awkcmd="awk  -v p=%s '{if($1 ~ /^-/ && $9 ~ p){print $9;exit(0)}}'"%pattern
                #proc     = subprocess.Popen(shlex.split(findcmd),stdout=subprocess.PIPE)
                ##awkproc  = subprocess.Popen(shlex.split(awkcmd),stdin=proc.stdout,stdout=subprocess.PIPE)
                ##proc.wait()
                #line = proc.stdout.readlines()
                ##print line,
                #lineLen = len(line)
                #if lineLen > 0:
                    #dictKey=os.path.basename(onedir)
                    #value=""
                    #for j in range(0,lineLen):
                        #value=value+os.path.basename(string.strip(line[j]))+","
    #                ondDict[dictKey]=value[0:len(value)-1]
        #break

    if out =="":
        for item in ondDict:
            print '%s,%s'%(item,ondDict[item])
    else:
        outfile=open(out,'w')
        for item in ondDict:
            print >> outfile,'%s,%s'%(item,ondDict[item])
        outfile.close()

def bookingGlobalDistribution(dirname,out):
    #if not os.path.isdir(dirname):
        #print "Error! %s is not a valid directory path"%dirname
        #sys.exit(2)
    if out == "":
        print "Error! please specify an output file name for global distribution"
        sys.exit(2)

    if _UPDATE or (not os.path.exists(out)):
        files = os.listdir(dirname)
        sortedfiles = sorted(files)
        ondDict = dict()
        for onebasefile in sortedfiles:
            onefile=os.path.join(dirname,onebasefile)
            f=open(onefile,'r')
            for line in f:
                strs=string.split(line,",")
                if strs[0] in ondDict:
                    ondDict[strs[0]] = int(strs[3])+ondDict[strs[0]]
                else:
                    ondDict[strs[0]] = int(strs[3])
                    if strs[0] == '1.0':
                        print onefile
                    elif strs[0] == '0.5':
                        print onefile
            f.close()
            #break

        sortedList = sorted(ondDict.iteritems(),key=operator.itemgetter(1),reverse=True)
        #if out =="":
            #for item in sortedList:
                #print '%s,%d'%(item[0],item[1])
        if out !="":
            outfile=open(out,'w')
            for item in sortedList:
                print >> outfile,'%s,%d'%(item[0],item[1])
            outfile.close()

    else:
        print "%s exists!"%out
    #printfromDictfile(out)





def bookingSummaryDistribution(dirname,_ond,_dds,num=-1,out=""):
    if not os.path.isdir(dirname):
        print "Error! %s is not a valid directory path"%dirname
        sys.exit(2)
    files = os.listdir(dirname)
    sortedfiles = sorted(files)
    n = len(string.split(_ond,"/"))
    daten = 6*n+n-1
    ondDict = dict()
    for onefile in sortedfiles:
        onefile=os.path.join(dirname,onefile)
        #onefile = '/remote/oridatacenter/PROCESSED/prorating_ond/summary_stats/od_dirc_dt_Ntkt_Ndt_Ntottkt_130415.csv'
        #print onefile
        awkcmd=[]
        awkcmd.append('awk')
        awkcmd.append('-v')
        awkcmd.append('ond=%s'%_ond)
        awkcmd.append('-v')
        awkcmd.append('dds=%s'%_dds)
        awkcmd.append('BEGIN{FS=","}{if($1 ~ "^"ond && $3 == dds) {print $3 "," $4 "," $0}}')
        awkcmd.append('%s'%onefile)

        #print awkcmd
        proc = subprocess.Popen(awkcmd,stdout=subprocess.PIPE)
        line = proc.stdout.readlines()
        lineLen = len(line)
        #print awkcmd
        for i in range(0,lineLen):
            onepair=string.split(line[i],",")
            dictKey=os.path.basename(onefile)
            if dictKey in ondDict:
                ondDict[dictKey]=int(onepair[1])+ondDict[dictKey]
            else:
                ondDict[dictKey]=int(onepair[1])
        #break

    sortedList = sorted(ondDict.iteritems(),key=operator.itemgetter(1),reverse=True)
    if out =="":
        for item in sortedList:
            print '%s,%d'%(item[0],item[1])
    else:
        print "Ready to out for bookingsummarydistribution"
        outfile=open(out,'w')
        for item in sortedList:
            print >> outfile,'%s,%d'%(item[0],item[1])
        outfile.close()

def bookingSummary(dirname,_ond,num=-1,out=""):
    if not os.path.isdir(dirname):
        print "Error! %s is not a valid directory path"%dirname
        sys.exit(2)
    files = os.listdir(dirname)
    sortedfiles = sorted(files)
    count = 0
    n = len(string.split(_ond,"/"))
    daten = 6*n+n-1
    ondDict = dict()
    for onefile in sortedfiles:
        #print onefile
        onefile=os.path.join(dirname,onefile)
        #onefile = '/remote/oridatacenter/PROCESSED/prorating_ond/summary_stats/od_dirc_dt_Ntkt_Ndt_Ntottkt_130530.csv'
        #print onefile
        awkcmd=[]
        awkcmd.append('awk')
        awkcmd.append('-v')
        awkcmd.append('ond=%s'%_ond)
        awkcmd.append('BEGIN{FS=","}{if($1 ~ "^"ond) {print $3 "," $4}}')
        awkcmd.append('%s'%onefile)

        proc = subprocess.Popen(awkcmd,stdout=subprocess.PIPE)
        line = proc.stdout.readlines()
        lineLen = len(line)
        #print awkcmd
        #keyDict = dict()
        for i in range(0,lineLen):
            onepair=string.split(line[i],",")
            dictKey=onepair[0][0:daten]
            if dictKey in ondDict:
                ondDict[dictKey]=int(onepair[1])+ondDict[dictKey]
            else:
                ondDict[dictKey]=int(onepair[1])
        #break

    sortedList = sorted(ondDict.iteritems(),key=operator.itemgetter(1),reverse=True)
    if out =="":
        for item in sortedList:
            print '%s,%d'%(item[0],item[1])
    else:
        outfile=open(out,'w')
        for item in sortedList:
            print >> outfile,'%s,%d'%(item[0],item[1])
        outfile.close()
        
        

    
"""
updateStat is used to update statistic informations of master pricer searches
In the mode for citypair, print the statistics of searches concerning one particular city pair, and print results to standard output
Or else update always updates the statsfile. the printList function determines where to output
num is only valide for citypair mode
"""
def updateStat(dirname, cityPair="",num= -1):
    #if outname == "":
        #f=sys.stdout
    #else:
        #f=open(outname,'w')
    dirs = os.listdir(dirname)
    dirList = []
    Onds = dict()
    #else:
        #count = 0;
        #cityPair = cityPair.upper()
    for onefile in dirs:
        onefile = os.path.join(dirname,onefile)
        if isdir(onefile):
            #print "directory: %s" % (onefile)
            dirList.append(onefile)
        #else:
            #print "file:%s" % (onefile)
    
    for onedir in dirList:
        files = os.listdir(onedir)
        for onefile in files:
            fileLen = len(onefile)
            indexBeg = 7
            od = ""
            while indexBeg <= (fileLen - 16): 
                origin = onefile[indexBeg:indexBeg+3]
                dest = onefile[indexBeg+3:indexBeg+6]
                od = origin+dest
                if cityPair == "":
                    if od in Onds:
                        Onds[od] = Onds[od]+1
                    else:
                        Onds[od] = 1
                else:
                    if od == cityPair:
                        filekey = os.path.basename(onedir)
                        if filekey in Onds:
                            Onds[filekey] = Onds[filekey]+1
                        else:
                            Onds[filekey] = 1
                indexBeg = indexBeg + 12
    
    if len(Onds) == 0:
        printBold("No result found!")
        return 
    else:
        printBold("Sorting...")
        sortedList = sorted(Onds.iteritems(),key=operator.itemgetter(1),reverse=True)
        if cityPair == "":
            printList(sortedList,-1,STATSFILE)
        else:
            #printSearch(sortedList,num)
            ondFile = os.path.join(STATSDIR,cityPair)
            printSearch(sortedList,num,ondFile)
            printBold('Statistic information of %s has been saved to  %s'%(cityPair,ondFile))

"""
Print origin and destination search statistics
by default print to standard output; 
print to a file is supported by specifying the outputfilename
the format is comma seperated tuple such as PAR,LON,329
"""
def printList( ondList, n = -1, outname=""):
    if outname == "":
        f=sys.stdout
    else:
        f=open(outname,'w')
    
    listLen = len(ondList)

    if ((n == -1) or (n > listLen)):# The case that n is not set from the command or n is too big
        n = listLen
    
    i = 0
    padLen = len(str(n))
    while i < n:
        origin = ondList[i][0][0:3]
        dest   = ondList[i][0][3:6]
        count  = ondList[i][1]
        print >> f, "%s,%s,%d" %(origin,dest,count)    

        i = i+1
    if outname == "":
        f.close()

'''
Another general print dictionary function
'''
def printfromDictfile(statinput,num=-1,isInt=True,ans=""):
    if not os.path.exists(statinput):
        logMsg = "Error: The input stat file %s does not exist!"%statinput
        print logMsg
        sys.exit(2)
    f=open(statinput,'r')
    tmpDict=dict()
    for line in f:
        fields=string.split(string.strip(line),",")
        if len(fields) > 1:
            if isInt:
                tmpDict[fields[0]]=int(fields[1])
            else:
                tmpDict[fields[0]]=fields[1]
    while True:
        if ans == "":
            if not _EXE:
                print "Which column would u like to order? (1: Date 2:Occurrence)"
                ans = raw_input()
            else:
                ans = "2"
        if ans=="1":
            sortedList = sorted(tmpDict.iteritems(),key=operator.itemgetter(0),reverse=True)
            break
        elif ans=="2":
            sortedList = sorted(tmpDict.iteritems(),key=operator.itemgetter(1),reverse=True)
            break
        else:
            print "Wrong Index",
            ans=""
            return

    if num== -1 or num > len(sortedList):
        num=len(sortedList)
    for i in range(0,num):
        if isInt:
            print "%s,%d"%(sortedList[i][0],sortedList[i][1])
        else:
            print "%s,%s"%(sortedList[i][0],sortedList[i][1])





"""
Another print function. similar as printList
Only difference is that it reads from the specific file
Always print to standard output
By default it prints all statistics.
Print concerning one specific origin and destination pair is supported by specifying ond argument
"""
def printfromStat( num = -1, ond = "",statinput =""):
    if statinput =="":
        f=open(STATSFILE,'r')
    elif myrexists(statinput):
        f=open(statinput,'r')
    else:
        logMsg = "Error: The input stat file %s does not exist!"%statinput
        sys.exit(2)
    if ond == "":
        printfromDictfile(statinput)
        #if num == -1:
            #for line in f:
                #print line,#with , at the end to prevent auto adding newline
            #f.close()
        #elif num >= 0:
            #i = 1
            #for line in f:
                #if i > num:
                    #break
                #print line,
                #i = i+1
        #else:
            #print start+"Error:"+end
            #print "       The number to display should be positive!!"
            #usage()
            #sys.exit(2)
    else:
        origin = ond[0:3].upper()
        dest   = ond[3:6].upper()
        find = False
        for line in f:
            cols = line.split(',')
            if cols[0] == origin and cols[1] == dest:
                print "%s => %s: %s" %(cols[0],cols[1],cols[2]),
                find = True
                break
        if not find:
            printBold("Sorry! we don\'t have information about this city pair: %s => %s" %(ond[0:3], ond[3:6]))
            f.close()
            return 0
    f.close()
    return 1


"""
Print search statistics for a particular city pair
By default print to standard output
Print to a file is supported by specifying the output filename
"""
def printSearch( ondList, n = -1,outname=""):
    if outname == "":
        f=sys.stdout
    else:
        f=open(outname,'w')

    listLen = len(ondList)

    if ((n == -1) or (n > listLen)):# The case that n is not set from the command or n is too big
        n = listLen

    i = 0
    while i < n:
        print >>f, "%s, %s" %(ondList[i][0],ondList[i][1])
        i = i+1

    if outname!= "":
        f.close()

"""
Interactive function for finding search statistics of a particular city pair
"""
def searchOnd( _ond ,_update=True,num = -1):    
    printBold("Do you want to look at the statistics of the search results of %s => %s (Y/y/N/n)?" %(_ond[0:3],_ond[3:6]))
    printBold("Attention: showing statistics takes a bit of time. 50s for example.")
    if not _EXE:
        ans = raw_input()
    else:
        ans ='y'
    if ans == "y" or ans == "Y":
        tmpstat=os.path.join(STATSDIR,_ond)
        if _update or (not myrexists(tmpstat)):
            updateStat( MPDAILYDIR, _ond,num)
        #else:
        printfromStat(num,"",tmpstat)

    
    ##Changed to call proDataProcessing.py
    #new part
    printBold("Do you want to proceed with prorating tickets information (Y/y/N/n)?")
    if not _EXE:
        ans = raw_input()
    else:
        ans ='n'
    #ans = "Y"
    if ans == "Y" or ans == "y":
        printBold("Which date would you like to extract the prorated tickets information?(YYYYMMJJ) e.g. 130412" )
        ans = raw_input()
        ans = string.strip(ans)
        pythoncmd = []
        pythoncmd.append('python')
        pythoncmd.append('%s'%PRODATASCRIPT)
        pythoncmd.append('-d')
        pythoncmd.append('%s'%ans)
        #if len(ans) !=6:
            #logMsg = START+"Error!: "+END+"Please respect date format strictly, eg. 130412"
            #print logMsg
            #print >>logfile,logMsg
            #sys.exit(2)
        #else:
            #monthDir ="20"+arg[0:2]+"-"+arg[2:4]
            #filename ="enrichedcoupon"+arg+".csv"
            #inputfile = os.path.join(REMOTEPATH,monthDir)
            #inputfile =os.path.join(inputfile,filename)
            #if not myrexists(inputfile):
                #logMsg = START+"ERROR!"+END+'The inputfile "%s" indicated by date:%s does not exist!'%(inputfile,arg)
                #print logMsg
                #print >>logfile,logMsg
                #sys.exit(2)
        printBold("Which columns would u like to output? Please indicate configuration file."+START+" (Press Enter for all columns)"+END)
        ans = raw_input()
        ans = string.strip(ans) 
        res = 2
        if len(ans) == 0:
            res = subprocess.call(pythoncmd) 
        else:
            pythoncmd.append('-s')
            pythoncmd.append('%s'%ans)
            res = subprocess.call(pythoncmd)
    else:
        sys.exit(0)

    ##To be continued while loop when date error occured
    #print "Which date would you like to extract the prorated tickets information?(YYYYMMJJ) e.g. 20130412"        
    #ans = raw_input()
    #month = ans[0:4]+"-"+ans[4:6]
    #dirMonth= os.path.join(PRORATING_OND,month)
    ##print dirMonth
    #dirMonths = os.listdir(PRORATING_OND)
    ##pprint.pprint(dirMonths)
    #if not month in dirMonths:
        #print START+"Error!"+END
        #print "The date: %s that you entered is not in good format or there is no prorating tieckets information for the date you entered" %ans
    #else:
        #filename = "enrichedcoupon"+ans[2:8]+".csv"
        #filepath = os.path.join(dirMonth,filename)
        #filenames = os.listdir(dirMonth)
        #if not filename in filenames:
            #print START+"Error!"+END
            #print "       The date: %s that you entered is not in good format or there is no prorating tieckets information for the date you entered" %ans
        #else:
            ##to be continued
            #print "Ok to call shell"
            #subprocess.call("%s %s" %ECEXTRACTSCRIPT,shell=True)

"""
Main function
"""
def main(argv):
    global _EXE
    global _UPDATE
    try:
        opts, args = getopt.getopt(argv,"hus:am:g:n:e",["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    num       = -1
    _display  = 1
    _num      = 0
    _search   = 0
    optCounts = 0
    _global   = 0
    _searchmp = 0

    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            sys.exit()
        elif opt in ("-u"):
            _UPDATE = True
        elif opt in ("-s"):
            _search   = 1
            _display  = 0
            _ond = arg
            optCounts = optCounts +1     
        elif opt in ("-a"):
            _display  = 1
            optCounts = optCounts +1     
        elif opt in ("-n"):
            try:
                _num      = 1
                num       = int(arg)
                #optCounts = optCounts +1     
            except ValueError:
                print START+"Error:"+END
                print "       Please input a number to display top shot city pairs.\n"
                usage()
                sys.exit(2)
        elif opt in ("-g"):
            _global = 1
            _ond    = arg
        elif opt in ("-m"):
            _searchmp = 1
            _ond    = arg
        elif opt in ("-e"):
            _EXE = True
        else:
            usage()
            sys.exit(2)

    init()
    #if optCounts > 1:
         #print START+"Error:"+END
        #print "       The options are mutually exclusive.\n"
        #usage()
        #sys.exit(2)
    if len(args) == 1:
        dirname = args[0] 
        dirname = os.path.abspath(dirname)
        if not os.path.exists(dirname):
            print START+"Error:"+END
            print "       The input directory does not exist: %s\n" %args[0]
            sys.exit(2)
        elif not os.path.isdir(dirname):
            print START+"Error:"+END
            print "       The input is not a directory: %s\n" %args[0]
            sys.exit(2)

    elif len(args) == 0:
        if _global:
            dirname = ECSUMMARY
        elif _searchmp:
            dirname = MPDAILYPRODIR
        else:
            dirname = MPDAILYDIR
    else:
        usage()
        sys.exit(2)
    if _global and _search:
        print " Not compatible with these two options: -g and -s"
        usage();
        sys.exit(2)

    if _UPDATE and (not _global) and (not _searchmp):
        updateStat(dirname)

    if _global == 1:
        start = time()
        _onds=string.split(_ond,"_")
        if _ond == ".":
            out=os.path.join(STATSDIR,'ECSummaryStats')
            bookingGlobalDistribution(dirname,out) 
            print "Saving results successfullt to %s"%out
            #bookingGlobalDistribution(dirname) 
        elif len(_onds) ==1: 
            out=os.path.join(STATSDIR,"ec_"+string.replace(_ond,"/","_"))
            if _UPDATE or (not os.path.exists(out)):
                bookingSummary(dirname,_ond,num,out)
            printfromDictfile(out,num)

        elif len(_onds) == 2:
            out=os.path.join(STATSDIR,"pro_"+string.replace(_ond,"/","_"))
            if _UPDATE or (not os.path.exists(out)):
                print "Called BookingSummaryDistribution"
                bookingSummaryDistribution(dirname,_onds[0],_onds[1],num,out)
            printfromDictfile(out,num)
        else:
            print "-g option Format Error! %s"%_ond
            sys.exit(2)
        exectime = time() - start
        logMsg ="Finished. Used %f secondes"%exectime 
        print "INFO: %s"%logMsg
        sys.exit(0)

    if _searchmp== 1:
        start = time()
        _onds=string.split(_ond,"_")
        if len(_onds) ==1: 
        #to be modified
            #res = printfromStat(num,_ond)
            _search = 1
        elif len(_onds) == 2:
            out=os.path.join(STATSDIR,"mp_"+string.replace(_ond,"/","_"))
            if _UPDATE or (not os.path.exists(out)):
                mpSummaryDistribution(dirname,_onds[0],_onds[1],num,out)
            printfromDictfile(out,num,False)
        else:
            print "-m option Format Error! %s"%_ond
            sys.exit(2)
        exectime = time() - start
        logMsg ="Finished. Used %f secondes"%exectime 
        print "INFO: %s"%logMsg
        sys.exit(0)

    if _search == 1:
        res = printfromStat(num,_ond)
        if res == 1:
        #Lazy modification
            #printBold("Do you want to proceed with searched city pair %s => %s (Y/y/N/n)?" %(_ond[0:3],_ond[3:6]))
            #ans = raw_input()
            ans = "Y"
            if ans == "y" or ans == "Y":
                searchOnd(_ond,_UPDATE,num)
            #else:
                #printBold("Do you want to proceed with other city pair (Y/y/N/n))?")
                #ans = raw_input()
                #if ans == "y" or ans == "Y":
                    #printBold("Please enter 3 letter code for origin city followed by 3 letter code of destination city. e.g. PARLON, which means Paris to London.")
                    #ans = raw_input().upper()[0:6]
                    #searchOnd(ans)
                #else:
                    #sys.exit(0)
        else:
            sys.exit(0)
    elif _display == 1 :
        printfromStat(num)
        if _EXE:
            print "In executation mode, no compatible"
            sys.exit(2)
            
        printBold("Do you want to proceed with one particular city pair (Y/y/N/n)?")
        ans = raw_input()
        if ans == "y" or ans == "Y":
            printBold("Please enter 3 letter code for origin city followed by 3 letter code of destination city. e.g. PARLON, which means Paris to London.")
            ans = raw_input().upper()[0:6]
            searchOnd(ans,_UPDATE,num)
        else:
            exit(0)
    else:
        sys.exit(2)

        




if __name__== '__main__':
    if len(sys.argv) == 1:
        print START+"ERROR!"+END
        print "       Missing arguments!"
        usage()
        sys.exit(2)
    main(sys.argv[1:])
    #print checkFile(sys.argv[1])
