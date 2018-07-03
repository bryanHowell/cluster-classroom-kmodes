'''
Created on Nov 1, 2017
@author Bryan Howell, Ph.D.

Description of procasmt module.
This module defines functions for processing assessments.
'''

import glob
import os
import numpy as np


def cumasmt_path(datadir, tchr, per, asmtdate):
    '''cumasmt_path determines the name and path to the cumulative assessment

    Parameters:
    datadir, the directory where all the educators' assessments are stored
    tchr, the teacher's name
    - format = [last name]_[middle initial]_[first name], x if no middle initial
    per, the (class) period of the course
    - format = periodx, where x = 1, 2, etc.
    asmtdate, the date of the formative or summative assessment
    - format = xxxxxx, month -> day -> year, 0x for single digits
    ** all letters are lower case and no spaces unless otherwise specified **

    Returns:
    casmt_dir, the path to the cumulative assessment
    casmt_file, the file name of the cumulative assessment (w/ path)
    '''

    casmt_dir = datadir+'/'+tchr+'/'+per+'/assessment_'+asmtdate  # get directory
    # get file name (w/ path)
    casmt_file = glob.glob(casmt_dir+'/cum_*.csv')[0]

    return casmt_dir, casmt_file


def asmt_files(datadir, tchr, per, asmtdate):
    '''asmt_paths find the file names of all assessments within a period

    Parameters:
    datadir, the directory where all the educators' assessments are stored
    tchr, the teacher's name
    - format = [last name]_[middle initial]_[first name], x if no middle initial
    per, the (class) period of the course
    - format = periodx, where x = 1, 2, etc.
    asmtdate, the date of the formative or summative assessment
    - format = xxxxxx, month -> day -> year, 0x for single digits
    ** all letters are lower case and no spaces unless otherwise specified **

    Returns:
    all_asmtfiles, a list of all the assessment files
    '''

    # read paths from parent, period directory
    workdir = os.getcwd()  # working directory
    perdir = datadir+'/'+tchr+'/'+per
    os.chdir(perdir)
    all_asmtdir = glob.glob('assessment*')  # sort by name
    os.chdir(workdir)
    # get dates and convert to integers
    numDir = len(all_asmtdir)
    intDates = [0] * numDir
    for ii in range(numDir):
        intPre = all_asmtdir[ii].split('_')[1]
        nn = len(intPre)
        intDates[ii] = int(intPre[(nn - 4):nn] + intPre[0:(nn - 4)])
    # sort dates
    sortIndx = sorted(range(numDir), key=lambda k: intDates[k])
    intDates = sorted(intDates)
    allAsmtDirSorted = [all_asmtdir[i] for i in sortIndx]
    # rearrange asmt date to match integer format
    nn = len(asmtdate)
    asmtDatev2 = int(asmtdate[(nn - 4):nn] + asmtdate[0:(nn - 4)])

    # find file names of assessments w/in paths
    all_asmtfiles = []  # preallocate
    for ii in range(numDir):
        # current directory and file
        tmp = glob.glob(perdir+'/'+allAsmtDirSorted[ii]+'/?_asmt*.csv')

        # break when current date is reached
        if intDates[ii] > asmtDatev2:
            break
        all_asmtfiles.append(tmp[0])

    return all_asmtfiles


def ckheader(alldata):
    '''checks the format of the assessment

    Parameters:
    alldata, a list of all the assessment data

    Returns:
    hdrows, number of rows in the header (of each assessment)
    '''

    nasmt = len(alldata)  # number of assessments
    hdrows = 4  # number of rows in header

    # check header of assessments
    goodasmt = 0
    for ii in range(nasmt):
        ckhd = (alldata[ii][0][0] == 'TEK') & (alldata[ii][1][0] == 'value') & (alldata[ii][2][0] == 'answer')\
            & (alldata[ii][3][0] == 'type')
        if(ckhd == True):
            goodasmt += 1
    if(goodasmt < nasmt):
        print('Error: one or more of the file headers has an incorrect format')
        exit()

    for ii in range(nasmt):
        for jj in range(hdrows):
            for kk in range(1, len(alldata[ii][jj])):
                cell_ijk = alldata[ii][jj][kk].strip()
                if(cell_ijk.isspace() or len(cell_ijk) < 1):  # if whitespace or empty
                    print('Error: one or more headers is missing data')
                    exit()

    return hdrows


def extract_data(datapre, hdrows):
    '''extract_data extracts data from an assessment

    Parameters:
    datapre, unprocessed data from a (formative or summative) assessment
    hdrows, number of rows in the header (of each assessment)

    Returns:
    d, dictionary holding extracted data
    d.tek0, parent TEK (e.g., parent of A.1 is A -> converted to 1)
    d.tek1, child TEK (e.g., child of A.1 is 1)
    d.ptvalue, point values of questions
    d.numkey, number key where A,B,... converted to 1,2,...
    d.anskey, answer key
    d.questype type of question
    - MC = multiple choice (ABCD or FGHJ)
    - GR = gridded response (numeric value)
    - FR = free response (partial credit from 0-X, where X is max. # of points)
    d.IDs, students identifiers
    d.data, graded assessment
    - MC and GR graded as right (1) or wrong (0)
    - FR graded as 0-X (see above for X)
    '''

    d = dict()  # preallocate dictionary

    # extract all information from header
    allinfo = [l[1:] for l in datapre[:hdrows]]  # TEKs, values, answers
    numq = len(allinfo[0])  # number of questions

    # TEK categories
    tekpre0 = allinfo[0]
    tekpre1 = [i.split('.', 1)[1] for i in tekpre0]
    d['tek0'] = [int(i.split('(', 1)[0]) for i in tekpre1]
    d['tek1'] = [ord(j.split(')', 1)[0]
                     )-64 for j in [i.split('(', 1)[1] for i in tekpre1]]

    # point values
    d['ptvalue'] = np.asarray([float(ii) for ii in allinfo[1]])

    # answer key
    key = allinfo[2]  # answer key
    numkey = np.zeros((numq))
    questype = allinfo[3]  # type of
    for ii in range(numq):
        if(key[ii] == 'A' or key[ii] == 'F'):
            numkey[ii] = 1
        elif(key[ii] == 'B' or key[ii] == 'G'):
            numkey[ii] = 2
        elif(key[ii] == 'C' or key[ii] == 'H'):
            numkey[ii] = 3
        elif(key[ii] == 'D' or key[ii] == 'J'):
            numkey[ii] = 4
        else:
            if(questype[ii] == 'FR'):
                numkey[ii] = key[ii]

    d['numkey'] = numkey
    d['anskey'] = key
    d['questype'] = questype

    # extract students' IDs
    d['IDs'] = [l[0] for l in datapre[hdrows:]]  # get student ids

    # extract answers and grade assessment
    X0 = [l[1:] for l in datapre[hdrows:]]  # get raw data
    # remove '+' from mc answers
    X1 = [[l[ii].replace('+', '').strip() if(questype[ii] == 'MC')
           else l[ii] for ii in range(numq)] for l in X0]

    # grading answers...
    d['data'] = grade_asmt(X1, questype, key)

    return d


def grade_asmt(asmt, qtype, akey):
    '''grade_asmt grades an assessment given question types and an answer key

    Parameters:
    asmt, the assessment
    - each row contains the answers for a student
    - each column contains a different question
    qtype, the type of question (MC, GR, or FR)
    - MC = multiple choice, GR = gridded response, FR = free response
    akey, the answer key
    - ABCD or FGHJ for MC number for GR 0-max. # for FR

    Returns:
    asmt_g, the graded assessment
    - right (1) or wrong (0) for MC and GR
    - 0-max. # for FR
    '''

    nstud = len(asmt)
    nques = len(qtype)  # number of questions
    asmt_g = [[0*rowi]*nques for rowi in range(nstud)]  # preallocate data

    for jj in range(nques):
        if(qtype[jj] == 'MC'or qtype[jj] == 'GR'):  # mult. choice/gridded response
            for ii in range(nstud):
                #                 print(akey[jj],asmt[ii][jj],akey[jj]==asmt[ii][jj])
                asmt_g[ii][jj] = int(akey[jj] == asmt[ii][jj])
        elif(qtype[jj] == 'FR'):  # free response
            for ii in range(nstud):
                if(asmt[ii][jj] != ''):
                    asmt_g[ii][jj] = float(asmt[ii][jj]) if (
                        '.' in asmt[ii][jj]) else int(asmt[ii][jj])
        else:
            print('Unknown type of question')
            exit()

    return asmt_g


def parse_quest(qtype):
    '''parse_quest parses questions into the different types

    Parameters:
    qtype, type of question 
    - MC = multiple choice, GR = gridded response, FR = free response

    Returns:
    qindx, dictionary containing indices of different types of questions
    '''

    # this is an inefficient way of getting indices (update later)
    qindx = dict()  # preallocate dictionary
    qindx['mc'] = np.where([xx == 'MC' for xx in qtype])[0]
    qindx['gr'] = np.where([xx == 'GR' for xx in qtype])[0]
    qindx['fr'] = np.where([xx == 'FR' for xx in qtype])[0]

    return qindx


def studmvt(Xall, hdrows):
    '''studmvt determines the students that moved across the assessments

    Parameters:
    Xall, data from all assessments (list)
    hdrows, number of rows in the header (of each assessment)

    Returns:
    Xall_id, IDs in each assessment (list)
    dset_idmvt, IDs of students that moved btwn assessments (dictionary of sets)
    - ID_all, unqiue IDs of all students still present at latest assessment (set)
    - ID_enter, unqiue IDs of all students the entered throughout assessments (set)
    - ID_exit, unique IDs of all student that exited throughout assessments (set)
    '''

    dset_idmvt = dict()  # preallocate dictionary
    nasmt = len(Xall)  # number of assessments

    # extract the 1st column from all assessments
    Xall_col0 = [[ll[0] for ll in l] for l in Xall]

    # get IDs from 1st column
    Xall_id = [l[hdrows:] for l in Xall_col0]  # list
    Xall_idsets = [set(l) for l in Xall_id]  # convert list to sets

    # determine all unique members across all sets
    Xall_allids = Xall_idsets[0].copy()  # starting set = initial set
    Xall_exitids = set()  # starting set = empty
    Xall_enterids = set()  # starting set = empty
    for xx in range(1, nasmt):
        # new members added to list
        Xall_allids.update(Xall_idsets[xx]-Xall_idsets[0])
        # members that entered and exited between two consecutive assessments
        Xall_exitids.update(Xall_idsets[xx-1]-Xall_idsets[xx])
        Xall_enterids.update(Xall_idsets[xx]-Xall_idsets[xx-1])

    # remove members that left
    Xall_allids = Xall_allids-(Xall_exitids-Xall_enterids)

    dset_idmvt['ID_all'] = Xall_allids
    dset_idmvt['ID_enter'] = Xall_enterids
    dset_idmvt['ID_exit'] = Xall_exitids
    dset_idmvt['ID_enter_now'] = Xall_idsets[nasmt-1]-Xall_idsets[nasmt-2]
    dset_idmvt['ID_exit_now'] = Xall_idsets[nasmt-2]-Xall_idsets[nasmt-1]

    return Xall_id, dset_idmvt


def reorg_asmt(Xall, hdrows, asmt_id, allid):
    '''reorg_asmt reorganizes assessments keeping only extant students

    Parameters:
    Xall, data from all assessments (list)
    hdrows, number of rows in the header (for each assessment)
    asmt_id, IDs from all assessments (list)
    allid, all unique IDs across all assessments (set)

    Returns:
    Xall_p, reorganized data from all assessments (list)
    '''

    nasmt = len(Xall)
    maxmem = len(allid)  # number of extant students across all assessments

    # convert IDs to integers
    int_asmtid = [[int(ll) for ll in l] for l in asmt_id]
    # sorting speeds up mapping (below)
    int_allid = sorted([int(s) for s in allid])

    # calculate mapping from each assessment to the reorganized format
    # in1d, see if one array is present in another
    # searchsorted, find indices where elements should be inserted to maintain order
    # asmtid_ref, IDs to go in reorganized assessments
    # asmtid_tar, target location of ID in reorganized assessment
    asmtid_ref = [np.nonzero(np.in1d(x, int_allid))[0] for x in int_asmtid]
    asmtid_tar = [np.searchsorted(int_allid, x) for x in int_asmtid]

    # transfer headers
    # preallocate cum. assessment
    Xall_p = [[[] for jj in range(maxmem+hdrows)] for ii in range(nasmt)]
    for ii in range(nasmt):  # for each assessment
        Xall_p[ii][0:hdrows] = Xall[ii][0:hdrows]  # copy header

    # transfer data
    for ii in range(nasmt):
        # only for members in cumulative assess.
        for jj in range(len(asmtid_ref[ii])):
            jjeff = asmtid_ref[ii][jj]
            # - NOTE: "numhd" rows are reserved for header, so include in mapping
            rpre = jjeff+hdrows  # row in current assess.
            rpost = asmtid_tar[ii][jjeff]+hdrows  # row in cumulative assess.
            Xall_p[ii][rpost] = Xall[ii][rpre]

    # fill in empty rows of cum. assessment
    for ii in range(nasmt):
        nrow = len(Xall_p[ii])
        ncol = len(Xall_p[ii][0])
        for jj in range(nrow):
            if(len(Xall_p[ii][jj]) < 1):
                Xall_p[ii][jj] = ['' if xx != 0 else str(
                    int_allid[jj-hdrows]) for xx in range(ncol)]

    return Xall_p


def combasmt(Xall):
    '''combasmt combines all assessments into one cumulative assessment

    Parameters:
    Xall, data from all assessments

    Returns:
    Xcum, data concatentated across all dates
    '''

    # merging files into 1
    Xcum = []  # preallocate
    for ii in range(len(Xall)):
        c0 = 0 if ii == 0 else 1
        tmp = list(map(list, zip(*Xall[ii])))  # transpose current list
        Xcum.extend(tmp[c0:])  # accumulate lists
    Xcum = list(map(list, zip(*Xcum)))  # transpose (back) total list

    return Xcum
