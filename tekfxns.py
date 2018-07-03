'''
Created on Nov 1, 2017
@author Bryan Howell, Ph.D.

Description of procasmt module.
This module defines functions for processing TEKS.
'''

import numpy as np


def four_binmetrics(x1, x2):
    '''four_binmetrics takes 2 binary vectors and returns the 4 standard metrics for their pairing

    Parameters:
    x1, binary vector 1 (contains 0s and 1s)
    x2, binary vector 2

    Returns:
    [s11, s10, s01, s00]
    - s11, number of times 1s match
    - s10, number of times 0 in x1 is paired w/ 1 in x2
    - s01, number of times 1 in x1 is paired w/ 0 in x2
    - s00, number of times 0s match
    '''

    n = len(x1)
    s11 = (x1*x2).sum()/n
    s10 = sum([int(bval) for bval in x1 > x2])/n
    s01 = sum([int(bval) for bval in x1 < x2])/n
    s00 = ((1-x1)*(1-x2)).sum()/n
    return [s11, s10, s01, s00]


def binsim(sxy):
    '''bin_similarity calculates similarity based on 4 standard metrics for binary vectors

    Parameters:
    sxy, vector containing s11, s10, s01, and s00 (in that order)

    Returns:
    S, similarity value (scalar)
    - based on Rogers-Tanmoto
    '''

    if(len(sxy) != 4):
        print('requires four binary metrics')
        exit()
    S = (sxy[0]+sxy[3])/(sxy[0]+sxy[3]+2*(sxy[1]+sxy[2]))
    return S


def numberteks(course):
    '''numberteks returns the number of teks for a given course

    Parameters:
    course, the subject being taught by the educator (algebra1, algebra2, or precal)

    Returns:
    teks0_course, the number of parent teks for a given course
    teks1_course, the number of child teks per parent tek
    '''

    teks0_course = {'algebra1': 12, 'algebra2': 8, 'precal': 5}  # parent teks
    teks1_course = {'algebra1': [7, 9, 8, 3, 3, 3, 3, 2, 5, 6, 2, 5], 'algebra2': [7, 4, 7, 8, 5, 12, 9, 3],
                    'precal': [7, 16, 9, 11, 14]}  # child teks

    return teks0_course.get(course), teks1_course.get(course)


def crseteks_intmap(n_tekc):
    '''crseteks_intmap constructs mapping from each tek to a unique integer

    Parameters:
    n_tekc, number of child TEKS per parent TEK category

    Returns:
    teks2int, mapping from tek to integer
    '''

    tot_tekc = sum(n_tekc)  # total number of child TEKS across all parent TEKs
    tot_tekp = len(n_tekc)  # total number of parent TEKS
    teks2int = [0]*tot_tekc  # preallocate

    cc = 0  # counter
    for ii in range(tot_tekp):  # for each parent category (px)
        tbase = 100*(ii+1)+1  # p1 -> 1xx, p2 -> 2xx, etc.
        teks2int[cc:(cc+n_tekc[ii])] = list(range(tbase, tbase+n_tekc[ii]))
        cc = cc+n_tekc[ii]

    return teks2int


def tek2int(tekp, tekc):
    '''tek2int assigns a unique integer for a given TEKS

    Parameters:
    tekp, parent TEKS number (A=1, B=2, etc.)
    tekc, child TEKS number

    Returns:
    uint, a unique integer    
    '''

    uint = np.add(tekp*100, tekc)  # assign each TEK a unique integer

    return uint


def accteks(X, teks, teks_crse, qtype, qval, nkey):
    '''accteks calculates the accuracy of TEKS for each student

    Parameters:
    X, assessment (# students x # questions)
    teks, TEKS for the assessment (1 x # questions)
    teks_crse, TEKS for the course (1 x total # of TEKS)
    qtype, type of question (MR, FR, or GR)
    qval, points value assigned to each question
    nkey, answers expressed as numbers
    - MR: A=1, B=2, etc.
    - FR: max. number of points
    - GR: answer

    Returns:
    Au, new matrix containing accuracy of each unique TEKS
    utek, unique TEKS ordered from least to greatest
    utek_map, map from TEKS for asmt to TEKS for course
    '''

    # organize and sort TEKs
    tek_si = np.argsort(teks)  # indices to sort TEKs
    teks = teks[tek_si]  # sort TEKs
    X = X[:, tek_si]  # sort data matrix
    qtype = qtype[tek_si]
    qval = qval[tek_si]
    nkey = nkey[tek_si]

    # determine which TEKS are unique
    # utek, unique TEKS
    # utek_i0, index where unique TEKS first appears in teks
    # utek_cnt, count of number of TEKS for a unique TEKS
    # utek_map, maps utek in asmt to utek in course list
    utek, utek_i0, utek_cnt = np.unique(
        teks, return_index=True, return_counts=True)
    utek_map = np.searchsorted(teks_crse, utek)
    m = X.shape[0]  # number of rows

    # consolidate raw data (w/ repeat TEKS) into new data matrix (w/ unique TEKS)
    Au = np.zeros((m, len(utek)))  # data matrix for unique TEKs
    for ii in range(len(utek)):
        i1 = utek_i0[ii]  # inclusive
        i2 = utek_i0[ii]+utek_cnt[ii]  # not inclusive
        for jj in range(i1, i2):
            if(qtype[jj] == 'FR'):
                Au[:, ii] = Au[:, ii]+X[:, jj] / \
                    np.full((m), nkey[jj])*np.full((m), qval[jj])
            else:
                Au[:, ii] = Au[:, ii]+X[:, jj]*np.full((m), qval[jj])
        Au[:, ii] = Au[:, ii]/np.full((m), np.sum(qval[i1:i2]))

    return Au, utek, utek_map


def binmat(M, vth):
    '''binmat binarizes a given matrix based on threshold

    Parameters:
    M, matrix to be binarized
    vth, threshold for binarization
    - 1 = value > vth
    - 0 otherwise

    Returns:
    Mbin, binarized matrix
    '''

    Mbin = np.asarray([[int(cij > vth) for cij in ri] for ri in M])

    return Mbin


def simteks(Btek, totteks, loc2glob, globtek):
    '''simteks quantifies similarity of TEKS

    Parameters:
    Btek, binary matrix (# students x # unique TEKS for asmt)
    - (mastery = 1, non-mastery = 0)
    totteks, total number of TEKS for a course
    loc2glob, is mapping from local TEKS (for asmt) to global TEKS (for course)
    globtek, all teks for a course expressed as unique integers

    Returns:
    D, matrix showing similarity between pairs of TEKS
    T2T, shows TEKS ranked from most to least similar for a given TEKS
    '''

    # dissimilarity (D) matrix
    # - perfectly dissimilar (1) by default
    D = np.ones((totteks, totteks))
    Dth = 0.5  # anything > 0.5 is purely due to chance
    # pairs TEKS based on D
    T2T = np.zeros((totteks, totteks))

    # calculating dissimilarity matrix
    for ii in range(totteks):
        D[ii, ii] = 0  # a given TEKS is perfectly similar (D=0) to itself

    for ii in range(Btek.shape[1]):
        for jj in range(ii+1, Btek.shape[1]):
            sfour = four_binmetrics(Btek[:, ii], Btek[:, jj])
            imap = loc2glob[ii]
            jmap = loc2glob[jj]
            dij = 1-binsim(sfour)
            if(dij < Dth):
                D[imap, jmap] = dij
                D[jmap, imap] = dij  # due to symmetry

    for ii in range(Btek.shape[1]):
        imap = loc2glob[ii]
        Drank_si = np.argsort(D[imap, :])
        T2T[imap] = np.asarray(globtek)[Drank_si]

    return D, T2T
