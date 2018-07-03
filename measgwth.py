'''
Created on Nov 1, 2017
@author Bryan Howell, Ph.D.

Description of procasmt module.
This module defines functions for measuring student growth.

Notes:
* currently coded for a two-parameter model w/ dichotomous data (0 or 1)
* will be generalized to a one-, two-, or three-parameter model
* will be generalized to polytomous data (0,1,2,...)
'''

import numpy as np
from math import log


def mapc_2vec(a, b):
    '''mapc_2vec calculates the max. absolute percent change between two vectors

    Parameters:
    a, vector/scalar 1
    b, vector/scalar 2

    Returns:
    mapc, maximum absolute percent change (w/r to b)
    '''

    return 100*np.amax(np.absolute((a-b)/b))


def stdvar(x):
    '''stdvar standarizes variable(s) w/r to its mean and standard deviation

    Parameters:
    x, input vector

    Returns:
    xs, x standardized
    xbar, mean of x
    sdx, standard deviation of x
    '''

    xbar = np.mean(x)
    sdx = np.std(x)

    return (x-xbar)/sdx, xbar, sdx


def Prob(xi, lam, th):
    '''Prob calculates probability of logistic equation given parameters

    Parameters:
    xi, intercept of z term
    lam, slope of z term    
    th, ability level
    * z = xi + lam*th

    Returns:
    P, probability of logistic eqn given parameters
    * P = 1/(1 + exp(-z))
    * z = xi + lam*th
    * lam = a, xi = -a*b     
    '''

    return 1 / (1 + np.exp(-(xi+lam*th)))


def logL(R, f, nu, th):
    '''logL is the log likelihood (L)

    Parameters:
    R, number of questions right (# groups x # items)
    f, number of subjects per group (# groups x 1)
    nu, parameters for item/question (# items x 1)
    th, abilities for all groups (# groups x 1)

    Returns:
    L, log likelihood (scalar)
    * L = co + sum_ij(rij*log(Pij),i=1,...,n,j=1,...,N_group)...
             + sum_ij((fj-rij)*log(1-Pij),i=1,...,n,j=1,...,N_group)
    * the constant, co, is arbitrary => ignored
    '''

    L = 0  # initialize
    for jj in range(R.shape[0]):  # for each subject
        for ii in range(R.shape[1]):  # for each item
            Pij = Prob(nu[0, ii], nu[1, ii], th[jj])
            if(Pij == 1):
                L += R[jj, ii]*log(Pij)
            else:
                L += R[jj, ii]*log(Pij)+(f[jj]-R[jj, ii])*log(1-Pij)

    return L


def gradL_th(r, f, P, lam):
    '''gradL_th calculates the gradient of log likelihood (L) w/r to ability

    Parameters:
    r, the number of right answers for all items (vector)
    f, the total number of questions for a subject (scalar)
    P, the true probability for all items of a given ability (vector)
    lam, rate parametere for all items (vector)

    Returns:
    dL/dth, rate of change of L w/r to ability (i.e., theta)
    * dL/dth = sum(lami*f*(ri/f-Pi),i=1,...,n)
    '''

    return np.sum(lam*f*(r/f-P))


def gradL_param(r, f, P, th):
    '''gradL_param calculates the gradient of log likelihood w/r to parameters

    Parameters:
    r, the number of right answers
    f, the total number of questions
    P, the true probability (given parameters)
    th, the ability of subject

    Returns:
    gradL = [L1, L2]', where ' = transpose operator
    * L1 = dL/d_xi = sum(fj*(rj/fj-Pj),j=1,...,N)
    * L2 = dL/d_lam = sum(fj*(rj/fj-Pj)*thj,j=1,...,N) 
    '''

    return np.array([np.sum(f*(r/f-P)), np.sum(f*(r/f-P)*th)])


def hessL_th(f, P, lam):
    '''gradL_th calculates the gradient of log likelihood (L) w/r to ability

    Parameters:
    f, the total number of questions for a subject (scalar)
    P, the true probability for all items of a given ability (vector)
    lam, rate parametere for all items (vector)

    Returns:
    dL/dth^2, second derivative of L w/r to ability (i.e., theta)
    * dL/dth^2 = -sum(lami^2*f*Pi*(1-Pi),i=1,...,n)
    '''

    return -1*np.sum(lam*lam*f*P*(1-P))


def hessL_param(f, P, th):
    '''logmodel_hessL calculates the Hessian of log likelihood (L) w/r to parameters

    Parameters:
    f, the total number of questions
    P, the true probability (given parameters)
    th, the ability of subject    

    Returns:
    H, Hessian of L
    * H = [L11,L12L21,L22]
    * L11 = d2L/d_xi^2 = -sum(fj*Pj*(1-Pj),j=1,...,N)
    * L12 = d2L/d_xi d_lam = -sum(fj*Pj*(1-Pj)*thj,j=1,...,N)
    * L21 = d2L/d_lam d_xi = ""
    * L22 = d2L/d_lam^2 = -sum(fj*Pj*(1-Pj)*thj^2,j=1,...,N)     
    '''

    L11 = -1*np.sum(f*P*(1-P))
    L12 = -1*np.sum(f*P*(1-P)*th)
    L21 = L12
    L22 = -1*np.sum(f*P*(1-P)*th*th)

    return np.array([[L11, L12], [L21, L22]])


def fit_logmodel(Xraw):
    '''fit_logmodel fits a logistic model

    Parameters:
    Xraw, raw responses (only dichotomous data - see Notes)

    Returns:
    TBD, 
    '''

    # defining pertinent variables
    N = Xraw.shape[0]  # number of subjects
    n = Xraw.shape[1]  # number of items

    # reorganize subjects based on score
    # (row0 to rowN -> least to greatest score)
    rsum = np.sum(Xraw, axis=0)  # row sum
    csum = np.sum(Xraw, axis=1)  # column sum / score
    rawscore_si = np.argsort(csum)  # how to sort scores
    csum = csum[rawscore_si]
    Xraw = Xraw[rawscore_si, :]

    # determine which rows and columns to keep
    col2keep = np.where([~(a or b) for a, b in zip(rsum == 0, rsum == N)])[0]
    row2keep = np.where([~(a or b) for a, b in zip(csum == 0, csum == n)])[0]
    Xp1 = Xraw[row2keep, :]
    Xp1 = Xp1[:, col2keep]

    # subjects grouped based on similar ability
    # (assume those w/ same scores have the same ability)
    uscore, uscore_i0, uscore_cnt = np.unique(
        csum[row2keep], return_index=True, return_counts=True)
    Nsub = len(uscore)
    nsub = len(col2keep)
    Xsub = np.zeros((Nsub, nsub))
    for ii in range(Nsub):
        Xsub[ii, :] = np.sum(
            Xp1[list(range(uscore_i0[ii], uscore_i0[ii]+uscore_cnt[ii])), :], axis=0)

    # initial conditions
    th_star = stdvar(uscore)[0]  # activities for each group
    nu_star = np.concatenate(
        (np.zeros((1, nsub)), np.ones((1, nsub))), axis=0)  # nu = [xilam]

    # Birnbaum Paradigm (BP) to fit logistic equation
    maxitn = 20  # max. number of iterations
    mapc_L = 10e3
    mapcth = 1  # max. abs. percent change of < 1 %
    ncyc = 0  # number of cycles of BP
    while(ncyc < maxitn and mapc_L > mapcth):

        # previous log likelihood
        Lprev = logL(Xsub, uscore_cnt, nu_star, th_star)

        # for each item, estimate pairs of parameters given all abilities
        for ii in range(nsub):

            itn = 0
            mapc = 10e3
            while(itn < maxitn and mapc > mapcth):

                Pj = Prob(nu_star[0, ii], nu_star[1, ii], th_star)
                dL = gradL_param(Xsub[:, ii], uscore_cnt, Pj, th_star)
                H = hessL_param(uscore_cnt, Pj, th_star)
                Hi = np.linalg.inv(H)

                nu_prev = np.copy(nu_star[:, ii])
                nu_star[:, ii] = nu_star[:, ii]-np.matmul(Hi, dL)
                mapc = mapc_2vec(nu_prev, nu_star[:, ii])
                itn += 1

        # for each subject, estimate ability given all pairs of parameters
        for jj in range(Nsub):

            itn = 0
            mapc = 10e3
            while(itn < maxitn and mapc > mapcth):
                Pi = Prob(nu_star[0, :], nu_star[1, :], th_star[jj])
                dL = gradL_th(Xsub[jj, :], uscore_cnt[jj], Pi, nu_star[1, :])
                H = hessL_th(uscore_cnt[jj], Pi, nu_star[1, :])
                th_prev = np.copy(th_star[jj])
                th_star[jj] = th_star[jj]-(1/H)*dL
                mapc = mapc_2vec(th_prev, th_star[jj])
                itn += 1

        # anchor/standardize variables
        th_star, thbar, sdth = stdvar(th_star)
        nu_star[0, :] = nu_star[0, :]+nu_star[1, :]*thbar
        nu_star[1, :] = nu_star[1, :]*sdth

        # assess change in log likelihood
        Lnext = logL(Xsub, uscore_cnt, nu_star, th_star)  # next L
        mapc_L = mapc_2vec(Lprev, Lnext)  # max. abs. percent change in L
        ncyc += 1  # add 1 to cycle counter
        Lprev = Lnext  # update L

    # map groups back to subjects
    th_subj = np.zeros((N))-10
    for k in range(Nsub):
        isubj = row2keep[list(range(uscore_i0[k], uscore_i0[k]+uscore_cnt[k]))]
        th_subj[isubj] = th_star[k]

    print(th_subj)

    return 0
