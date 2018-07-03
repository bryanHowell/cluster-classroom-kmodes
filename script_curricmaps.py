'''
Created on Nov 15, 2017
@author Bryan Howell, Ph.D.

Description of script.
This script reads finds unique TEKS and determines how TEKS covary with each other
based on mastery across students.
'''

''' ========== Python Modules ========== '''

# base open-source python modules
import numpy as np
import matplotlib.pyplot as plt

# Bryan Howell's python modules
import csvwrap as csvw
import procasmt as pasmt
import tekfxns as tfxn


''' ==================== User Input ==================== '''

teacher = 'lewis_a_howell'
course = 'algebra2'
period = 'period5'
asmtdate = '11022017'
# datadir='/home/bryan/Desktop/edusoft_datatools/retrospec_dataset_v3'
datadir = '/home/bryan/Desktop/edusoft_datatools/ClusterDatasets'

asmtdir, asmtfile = pasmt.cumasmt_path(datadir, teacher, period, asmtdate)
asmtdata = csvw.read_csv(asmtfile)


''' ==================== TEKS for Course ==================== '''

num_tekc = tfxn.numberteks(course)[1]
tot_tekc = sum(num_tekc)
tot_tekp = len(num_tekc)

# create mapping from TEKs to integers
teks2int = tfxn.crseteks_intmap(num_tekc)


''' ========== Extract Data from Assessments ========== '''

numhd = pasmt.ckheader([asmtdata])
adata = pasmt.extract_data(asmtdata, numhd)

# extract data
X = np.array(adata.get('data'))  # data
nstud = X.shape[0]  # number of students
nques = X.shape[1]  # number of questions

# types of questions
qtype = adata.get('questype')
qindx = pasmt.parse_quest(qtype)

# value of questions
ptval = adata.get('ptvalue')  # value of each question
numkey = adata.get('numkey')  # answer key
keytype = np.asarray(adata.get('questype'))


''' ========== TEKS for Assessment ========== '''

# assign unique integer for eack TEKS
tek0 = np.asarray(adata.get('tek0'))
tek1 = np.asarray(adata.get('tek1'))
tek_asmt = tfxn.tek2int(tek0, tek1)


''' ========== Calculate Similarity of TEKS ========== '''

# get accuracy of TEKS
UTacc, utek_asmt, utek_asmt2crse = tfxn.accteks(
    X, tek_asmt, teks2int, keytype, ptval, numkey)
# binarize accuracy matrix based on mastery threshold
UTbin = tfxn.binmat(UTacc, 0.5)
# determine similarity btwn TEKS
Dtek, tek2tek = tfxn.simteks(UTbin, tot_tekc, utek_asmt2crse, teks2int)

print(tek2tek[:, :10])

''' ========== Visualize Results ========== '''

f, ax2 = plt.subplots()
# ax1.imshow(Xu)
# ax1.set_xlabel('Category',fontsize=30)
# ax1.set_ylabel('Student',fontsize=30)
# ax1.axis('off')
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
ax2.imshow(Dtek)
ax2.set_xlabel('Category', fontsize=30)
ax2.set_ylabel('Category', fontsize=30)
ax2.axis('off')
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
plt.show()
