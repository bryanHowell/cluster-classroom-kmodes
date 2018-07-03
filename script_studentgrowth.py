'''
Created on Nov 15, 2017
@author Bryan Howell, Ph.D.

Description of script.
This script uses structural equation modeling (SEM) and item response theory (IRT) to
quantify latent growth and ability.
'''

''' ========== Python Modules ========== '''

# base open-source python modules
import numpy as np
import time
# import matplotlib.pyplot as plt

# Bryan Howell's python modules
import csvwrap as csvw
import procasmt as pasmt
import measgwth as msg

''' ==================== User Input ==================== '''

teacher = 'lewis_a_howell'
course = 'algebra2'
period = 'period5'
asmtdate = '11022017'
# datadir='/home/bryan/Desktop/edusoft_datatools/retrospec_dataset_v3'
datadir = '/home/bryan/Desktop/edusoft_datatools/ClusterDatasets'

asmtdir, asmtfile = pasmt.cumasmt_path(datadir, teacher, period, asmtdate)
asmtdata = csvw.read_csv(asmtfile)

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
bdata_indx = np.sort(np.concatenate((qindx.get('mc'), qindx.get('gr'))))

# value of questions
ptval = adata.get('ptvalue')  # value of each question
numkey = adata.get('numkey')  # answer key
keytype = np.asarray(adata.get('questype'))

''' ========== Extract Data from Assessments ========== '''

to = time.time()
msg.fit_logmodel(X[:, bdata_indx])
tf = time.time()

print('Logistic model fitted in', tf-to, 's')
exit()
