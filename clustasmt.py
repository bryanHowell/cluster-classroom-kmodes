'''
Created on Nov 8, 2017
@author Bryan Howell, Ph.D.

Description of procasmt module.
This module defines functions for clustering assessments.
'''


# base python modules
import numpy as np
import math
import copy
import time
from kmodes import kmodes
import matplotlib.pyplot as plt


def gen_randasmt(numkey, mc_i, gr_i, fr_i, nstu, nque):
    '''generate_randassess constructs and grades an assessment w/ random answers

    Parameters:
    numkey, number key where A,B,... converted to 1,2,...
    x_i, indices of three different types of questions, x=mc,gr,fr
    - mc = multiple choice
    - gr = gridded response
    - fr = free response
    nstu, number of students
    nque, number of questions

    Returns:
    Xrand, graded assessment w/ random answers
    - mc, random guessing
    - gr, 50/50% chance of right/wrong
    - fr, points randomly awarded from 0-X, where X = max. # of points  
    '''

    # number of questions
    n_mcques = len(mc_i)  # number of multiple choice
    nchoice = 4  # number of choices (e.g., 4=[A,B,C,D])
    n_grques = len(gr_i)  # number of multiple choice
    n_frques = len(fr_i)  # number of free response

    # create assessment
    Xtmp = np.zeros((nstu, nque))  # preallocate

    # random guess for multiple choice (as integers: A=1, B=2, etc.)
    mc_guess = np.random.randint(nchoice, size=(nstu, n_mcques))+1
    # gr randomly graded as right (1) or wrong (0)
    gr_guess = np.random.randint(2, size=(nstu, n_grques))
    # random point assignment for free response
    fr_guess = np.zeros((nstu, n_frques))  # preallocate
    for ii in range(n_frques):
        cval = int(numkey[fr_i[ii]])+1
        fr_guess[:, ii] = np.random.randint(cval, size=(nstu))

    # combine guesses into a whole assessment
    Xtmp[:, mc_i] = mc_guess
    Xtmp[:, gr_i] = gr_guess
    Xtmp[:, fr_i] = fr_guess

    # grade the assessment
    Xrand = copy.deepcopy(Xtmp)  # pass by value (real copy)
    for jj in mc_i:
        for ii in range(nstu):
            Xrand[ii, jj] = int(Xtmp[ii, jj] == numkey[jj])

    return Xrand


def cluster_asmt(A, nclus, ntries):
    '''cluster_asmt clusters an assessment using kmodes algorithm

    Parameters:
    A, assessment (# students x # questions)
    nclus, number of clusters
    ntries, number of times to try kmodes algorithm

    Returns:
    c_indx, the index of the cluster to which the student is assigned
    c_cent, the centroid of each cluster
    c_distn, the total distortion (scalar) of the clustering
    '''

    km = kmodes.KModes(n_clusters=nclus, init='Huang',
                       n_init=ntries, verbose=0)
    c_indx = km.fit_predict(A)
    c_cent = km.cluster_centroids_
    c_distn = distn_kmode(A, c_indx, c_cent)

    return c_indx, c_cent, c_distn


def calc_optclus(X, qnkey, dic_qindx, maxclus, nrep, ntries):
    '''calc_optclus predicts optimal number of clusters

    Parameters:
    X, assessment (# students x # questions)
    dic_qindx, dictionary containing indices of each question type
    - MC = multiple choice, GR = gridded response, FR = free response
    qnkey, number key for questions
    - MC: A=1, B=2, etc.
    - FR: 0-max. # of points    
    maxclus, maximum number of clusters
    nrep, number of times to repeat each random assessment
    ntries, number of times to repeat kmodes algorithm

    Returns:
    optnum, optimal number of clusters
    '''

    # size of assessment
    nstud = X.shape[0]  # number of students
    nques = X.shape[1]  # number of questions
    # null and sample distortions
    Dsamp = np.zeros((maxclus, 1))
    Dnull = np.zeros((maxclus, nrep))
    # indices of each type of question
    mc_ix = dic_qindx.get('mc')
    gr_ix = dic_qindx.get('gr')
    fr_ix = dic_qindx.get('fr')

    # cluster and calculate distortions for null dist.
    print("clustering null dataset...")
    t0 = time.time()
    for bb in range(nrep):
        Xrnd = gen_randasmt(qnkey, mc_ix, gr_ix, fr_ix,
                            nstud, nques)  # random assessment
        for aa in range(maxclus):
            clus_D = cluster_asmt(Xrnd, aa+1, ntries)[2]
            Dnull[aa, bb] = clus_D
    t1 = time.time()
    print('Null dataset clustered in', t1-t0, 's')

    # cluster and calculate distortions for experimental dist.
    print("clustering experimental dataset...")
    t0 = time.time()
    for aa in range(maxclus):
        clus_D = cluster_asmt(X, aa+1, ntries)[2]
        Dsamp[aa] = clus_D
    t1 = time.time()
    print('Experimental dataset clustered in', t1-t0, 's')

    gapstats = gapstat(Dnull, Dsamp)

    print('Summary of gap statistics:')
    print(gapstats)
    posval = np.where(gapstats[:, 1] > 0)[0]+1
    optclus = posval[0]

    return optclus


def distn_kmode(X, c_indx, c_cent):
    '''distn_kmode calculates the kmodes distortion for a given clustering

    Parameters:
    X, data that was clustered (n x p)
    c_indx, cluster number (nc x p, where nc = # of clusters)
    c_cent, the centroid of each cluster (n x 1)

    Returns:
    D, the total distortion (scalar)
    1. sum all Hamming distances per group (across columns)
    2. reshape -> nc x 1 and divide by respect. number of entities (factor of 2 based on theory)
    3. total distortion is the sum of all local distortions    
    '''

    nc = c_cent.shape[0]  # number of clusters
    dim = X.shape  # dimensions of data (i.e., n x p)
    Dpre = np.zeros((nc, dim[0]))  # (each cluster has <= n entities)
    nk = np.zeros((nc, 1))  # preallocate # entities / cluster

    for ii in range(nc):  # for each cluster...

        ckindx = np.where(c_indx == ii)  # row indices for respect. cluster
        nindx = len(ckindx[0])  # number of indices
        nk[ii] = nindx  # number of indices = number of entities in cluster
        x2 = c_cent[ii, :]  # grap respect. cluster

        for jj in range(nindx):  # for each entity
            x1 = X[ckindx[0][jj], :]  # grap respect. row from data
            Dpre[ii, jj] = np.asarray(x1 != x2).astype(
                int).sum()  # calculatle Hamming distance

    Dlocal = np.reshape(Dpre.sum(axis=1), (nc, 1))/nk/2
    return Dlocal.sum()


def gapstat(dnull, dexp):
    '''gapstat calculates gap statistic

    Parameters:
    dnull, distortions from assessments w/ random answers (n_maxc x n_rpt)
    - n_maxc = maximum # of clusters
    - n_rpt = # of times to repeat each clustering
    dexp, distortions from assessments w/ real answers (n_maxc x 1)

    Returns:
    gapsumm, summaray of gap statistics (n_maxc x 2)
    - column 1 = number of clusters
    - column 2 = gap statistic
    '''

    # max. number of clusters
    ncmax = dnull.shape[0]
    # gapk = mean log distortion of null (across repeats) - log distortion of exp.
    gapk = np.mean(np.log(dnull), axis=1).reshape(ncmax, 1)-np.log(dexp)
    # sk = scaled standard deviation log distortion of null
    sk = np.std(np.log(dnull), axis=1).reshape(
        ncmax, 1)*math.sqrt(1+1/dnull.shape[1])
    # looking where g_k is greater than (g_(k+1) + sg_(k+1)), where k is # of clusters
    ckgap = gapk[0:(ncmax-1)]-(gapk[1:ncmax]-sk[1:ncmax])

    kindx = np.arange(ncmax-1).reshape((ncmax-1), 1) + \
        1  # cluster number (1,2,3,etc..)
    # col 1 = cluster indx, col 2 = gap stat.
    gapsumm = np.concatenate((kindx, ckgap), axis=1)

    return gapsumm


def tek_time(c_cent, dic_qindx, qval, qnkey, tek0, ntek0):
    '''tek_time calculates how much time to devote to each parent tek

    Parameters:
    c_cent, the centroid of each cluster (nclus x 1, where nclus = # of clusters)
    dic_qindx, dictionary containing indices of each question type
    - MC = multiple choice, GR = gridded response, FR = free response
    qval, point value assigned to each question (nquest x 1)
    qnkey, number key for questions
    - MC: A=1, B=2, etc.
    - GR: number answer
    - FR: 0-max. # of points
    tek0, parent tek for each question (nquest x 1)
    ntek0, number of parent teks (depends on course) 

    Returns:
    tektime, the time to devote to each parent tek, for each cluster
    - the times are normalized so that the total time is 1
    '''

    numc = c_cent.shape[0]  # number of clusters
    nques = c_cent.shape[1]  # number of questions
    # preallocate
    tektime = np.zeros((numc, ntek0))  # time to spend on teks
    ptsloss = np.zeros((numc, nques))  # points lost
    # organize indices
    mcgr_ix = np.concatenate((dic_qindx.get('mc'), dic_qindx.get('gr')))
    fr_ix = dic_qindx.get('fr')

    for k in range(numc):  # for each cluster...

        # points lost for questions
        ptsloss[k, mcgr_ix] = (1-c_cent[k, mcgr_ix])*qval[mcgr_ix]
        ptsloss[k, fr_ix] = qval[fr_ix]*(1-c_cent[k, fr_ix]/qnkey[fr_ix])

        for utek_i in np.unique(tek0):  # for each unqiue tek0
            allutek_i = np.where(tek0 == utek_i)[0]
            tektime[k, utek_i -
                    1] = np.sum(ptsloss[k, allutek_i])/np.sum(qval[allutek_i])

        tektime[k, :] = tektime[k, :]/np.sum(tektime[k, :])  # normalize values

    return tektime


def order_clus(c_cent, c_oldid, dic_qindx, qval, qnkey):
    '''order_clus orders clusters based on accuracy of group

    Parameters:
    c_cent, the centroid of each cluster (nclus x 1, where nclus = # of clusters)
    c_oldid, old/current cluster number (nc x p, where nc = # of clusters)    
    dic_qindx, dictionary containing indices of each question type
    - MC = multiple choice, GR = gridded response, FR = free response
    qval, point value assigned to each question (nquest x 1)
    qnkey, number key for questions
    - MC: A=1, B=2, etc.
    - GR: number answer
    - FR: 0-max. # of points

    Returns:
    c_newid, new ordering of cluster IDs from least to most accurate (% correct)
    c_ord, how to reorder n clusters from 1-n, where n is the most accurate cluster
    '''

    numc = c_cent.shape[0]  # number of clusters
    # preallocate
    percor = np.zeros((numc))  # percent correct
    totval = np.sum(qval)  # max. number of points
    # organize indices
    mcgr_ix = np.concatenate((dic_qindx.get('mc'), dic_qindx.get('gr')))
    fr_ix = dic_qindx.get('fr')

    # for each cluster...
    for k in range(numc):
        # calcuate percentage that is correct
        n1 = np.sum(c_cent[k, mcgr_ix]*qval[mcgr_ix])
        n2 = np.sum(c_cent[k, fr_ix]/qnkey[fr_ix]*qval[fr_ix])
        percor[k] = 100*(n1+n2)/totval

    # organize by percent correct
    c_ord = np.argsort(percor)
    c_newid = np.zeros(c_oldid.shape)
    for ii in range(numc):
        # which ID goes 1st, 2nd, etc.?
        # find all instances of this ID
        # reassign to 1, 2, etc..
        jj = np.where(c_oldid == c_ord[ii])[0]
        c_newid[jj] = ii

    return c_newid, c_ord


def plot_clus(X, c_indx):
    '''plot_clus visualizes the clustering

    Parameters:
    X, assessment (# students x # questions)
    c_indx, cluster number (nc x p, where nc = # of clusters)

    Returns:
    N/A
    '''

    # size of assessment
    nstud = X.shape[0]  # number of students
    # sort indicces
    sort_cindx = np.argsort(c_indx)

    XX = X[sort_cindx, :]  # sorted assessment
    I1 = np.concatenate((5*c_indx.reshape((nstud, 1))+5, X),
                        axis=1)  # image of unorganized asmt
    # "" organized asmt
    I2 = np.concatenate(
        (5*c_indx[sort_cindx].reshape((nstud, 1))+5, XX), axis=1)

    # plotting...
    plt.figure(1)
    ax1 = plt.subplot(121)
    ax1.imshow(I1)
    ax1.axis('off')
    ax2 = plt.subplot(122)
    ax2.imshow(I2)
    ax2.axis('off')
    plt.show()

    return 0
