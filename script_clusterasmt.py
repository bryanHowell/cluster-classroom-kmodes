'''
Created on Jun 13, 2017
@author Bryan Howell, Ph.D.

Description of script.
This script reads in a cumulative assessment and uses a kmodes clustering
algorithm to parse students into groups. The optimal number of clusters is
predicted using a gap statistic.
'''


''' ========== Python Modules ========== '''

# base open-source python modules
import numpy as np
import time

# Bryan Howell's python modules
import csvwrap as csvw
import procasmt as pasmt
import clustasmt as casmt
import tekfxns as tfxn

''' ========== User Input ========== '''

findopt = False  # True = calculate optimal #, False = provide # of clusters
numclus = 2  # optimal number of clusters (default=2)
savedata = False

teacher = 'very_cool_teacher'
course = 'algebra2'
period = 'period5'
asmtdate = '05012017'
datadir = '/home/bryan/Desktop/edusoft_datatools/clusterClassroom/retrospec_dataset_v3'

''' ========== Teacher Tag ========== '''

tmpName = teacher.split('_')
endTag = '_' + tmpName[0][0] + tmpName[1] + tmpName[2][0] + course[0:3] + course[-1] + \
    period[0:3] + period[-1]

''' ========== Assessment Data ========== '''

numteks = tfxn.numberteks(course)[0]

# read assessment and document its location
t0 = time.time()
asmtdir, asmtfile = pasmt.cumasmt_path(datadir, teacher, period, asmtdate)
asmtdata = csvw.read_csv(asmtfile)
numhd = pasmt.ckheader([asmtdata])
adata = pasmt.extract_data(asmtdata, numhd)
t1 = time.time()
print('Data extraction completed in', t1-t0, 's')

# extract data
X = np.array(adata.get('data'))  # data
nstud = X.shape[0]  # number of students
nques = X.shape[1]  # number of questions


''' ========== Extract Data from Assessments ========== '''

# types of questions
qtype = adata.get('questype')
qindx = pasmt.parse_quest(qtype)

# value of questions
ptval = adata.get('ptvalue')  # value of each question
numkey = adata.get('numkey')  # answer key

# teks for questions
tek0 = np.asarray(adata.get('tek0'))
keytype = np.asarray(adata.get('questype'))


''' ========== Cluster Assessment ========== '''

if(findopt):
    optclus = casmt.calc_optclus(X, numkey, qindx, 5, 50, 50)
    print('Optimal number of clusters is', optclus)
    exit()
else:
    print("clustering experimental dataset...")
    t0 = time.time()
    clus_indx, clus_cent, clus_D = casmt.cluster_asmt(X, numclus, 100)
    t1 = time.time()
    print('Experimental dataset clustered in', t1-t0, 's')


''' ========== Postprocessing ========== '''

# calculate TEK times
tektime = casmt.tek_time(clus_cent, qindx, ptval, numkey, tek0, numteks)
# student and cluster indices (reordered)
studids = np.asarray(adata.get('IDs'))
newclus_indx, clusord = casmt.order_clus(
    clus_cent, clus_indx, qindx, ptval, numkey)
idmap = np.concatenate(
    (studids.reshape(nstud, 1), newclus_indx.reshape(nstud, 1)), axis=1)
# reorder metrics for each cluster
clus_cent = clus_cent[clusord, :]
tektime = tektime[clusord, :]

print('Summary of Clustering:')
print(clus_cent)
print(tektime)


''' ========== Saving Data ========== '''

# ** will be moved to part w/ file saving **
tmp = asmtfile.split('_')
fntag = tmp[-3]+'_'+tmp[-2]
fout_pre = 'cluster_'


# writing file
if(savedata == True and findopt == False):
    fnout1 = asmtdir+'/' + fout_pre + 'ids_' + fntag + endTag + '.csv'
    fnout2 = asmtdir+'/' + fout_pre + 'centroids_' + fntag + endTag + '.csv'
    fnout3 = asmtdir+'/' + fout_pre + 'tek0time_' + fntag + endTag + '.csv'
    csvw.write_csv(idmap, fnout1)
    csvw.write_csv(clus_cent, fnout2)
    csvw.write_csv(tektime, fnout3)

casmt.plot_clus(X, newclus_indx)
