'''
Created on Sep 09, 2017

@author: Bryan Howell, Ph.D.
'''

'''
==================== CLASSES TO IMPORT ==================== 
'''
# import os.path
import os
import glob
import numpy as np
import csv
from kmodes import kmodes
import time
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# import copy
# import math

'''
====================  USER INPUT ==================== 
'''

datadir = '/home/bryan/Desktop/edusoft_datatools/retrospec_dataset_v2/lewis_a_howell/algebra2/period5/assessment_05012017'
# datadir='/home/bryan/Desktop/edusoft_datatools/clusterstudy_datasets/blanca_f_oropez/period4/assessment_09012017'

'''
==================== FXNS ==================== 
'''


def read_csv(fname):
    # read_csv takes in a file name and reads in csv file
    f = open(fname, 'r')
    reader = csv.reader(f)
    fdata = []
    for row in reader:
        fdata.append(row)
    f.close()
    return fdata


def extract_data(datapre):
    # extract_data takes in an assessment and extracts the following:
    # 1. info.= TEKs + value + answers (3 x n), where n = # of questions
    # 2. student IDs
    # 3. data (students' answers)
    # outputs a dictionary

    d = dict()  # preallocate dictionary

    # ckeck to see if info. in file is in the right format
    ckinfo = (datapre[0][0] == 'TEK') & (datapre[1][0]
                                         == 'value') & (datapre[2][0] == 'answer')
    if ckinfo == False:
        print('Bad format for assessment!')
        exit()

    # extract all information from header
    allinfo = [l[1:] for l in datapre[:3]]  # TEKs, values, answers
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
    key = allinfo[2]  # key
    keyint = [[ord(key[ii])-64 if key[ii].isalpha() else float(key[ii])]
              for ii in range(numq)]  # answers as integers (A=1, etc.)
    # multiple choice (true) or free response (false)
    keytype = [key[ii].isalpha() for ii in range(numq)]
    d['anskey'] = key
    d['intkey'] = np.asarray(keyint).reshape(numq)
    d['keytype'] = keytype

    # extract students' IDs
    d['IDs'] = [l[0] for l in datapre[3:]]  # get student ids

    # extract answers and grade assessment
    key = datapre[2][1:]  # answer key
    X0 = [l[1:] for l in datapre[3:]]  # get raw data
    X1 = [[w.replace('+', '').strip() for w in l]
          for l in X0]  # remove '+' before letters

    # grading answers...
    d['data'] = [[int(key[x] == l[x]) if key[x].isalpha() else (
        0 if l[x] == '' else int(float(l[x]))) for x in range(len(l))] for l in X1]

    return d


'''
====================  FILE INFORMATION ==================== 
This section is where file information is derived and read
'''

# directories
workdir = os.getcwd()  # working directory
# files
os.chdir(datadir)
datafile = datadir+'/'+glob.glob('cum_*data.csv')[0]
os.chdir(workdir)
# read data from file
dataraw = read_csv(datafile)


'''
====================  PREPARE DATA ==================== 
This section is where file information is read and processed
'''

t0 = time.time()
adata = extract_data(dataraw)
t1 = time.time()
print('Data extraction completed in', t1-t0, 's')

# data to cluster
X = np.array(adata.get('data'))  # data
nstud = X.shape[0]  # number of students
nques = X.shape[1]  # number of questions

ikey = adata.get('intkey')  # answer key
mc_ix = np.where(adata.get('keytype'))[0]  # multiple choice questions
fr_ix = np.where(np.invert(adata.get('keytype')))[0]  # free response questions
ptval = adata.get('ptvalue')  # value of each question

tek0 = np.asarray(adata.get('tek0'))
keytype = np.asarray(adata.get('keytype'))

'''
====================  Principle Component Analysis ==================== 
'''

Xw = preprocessing.scale(X)  # whiten the data (0-mean, unit variance)
pca = PCA(n_components=2)
pca.fit(Xw)
vpc = np.transpose(pca.components_)  # principal axes
Z = np.dot(X, vpc)  # transform data to PC space


'''
====================  CLUSTERING DATA ==================== 
This section is where the data is clustered. For an exploratory analysis, clustering
is conducted on both random and experimental datasets. Based on a gap statistic, the
optimal number of clustered is predicted (first time gap stat. > 0). For a confirmatory
analysis, only the experimental data is clustered based on the optimal number of clusters.

'''

ntries = 100  # number of tries to kmodes algorithm

print("clustering experimental dataset...")
t0 = time.time()
km = kmodes.KModes(n_clusters=2, init='Huang', n_init=ntries, verbose=0)
clusters = km.fit_predict(X)
t1 = time.time()
print('Experimental dataset clustered in', t1-t0, 's')

'''
====================  Visualize Clusters ==================== 
'''

ix1 = clusters == 0
ix2 = clusters == 1

fig, ax = plt.subplots()
h1 = plt.scatter(Z[ix1, 0], Z[ix1, 1], color='k', s=3e3, marker='.')
h2 = plt.scatter(Z[ix2, 0], Z[ix2, 1], color='g', s=3e3, marker='.')
# plt.xlabel('PC1',fontsize=30)
# plt.ylabel('PC2',fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
# plt.legend((h1,h2),('group 1','group 2'),fontsize=20)
# ax.axis('off')
plt.show()

exit()
