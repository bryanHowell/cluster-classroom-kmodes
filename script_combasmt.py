'''
Created on Jun 13, 2017
@author Bryan Howell, Ph.D.

Description of script.
This script reads all formative and summative assessments for a given period in a course
and concatenates all the data.
'''

''' ========== Python Modules ========== '''

# stock and open-source python modules
import os.path

# Bryan Howell's python modules
import csvwrap as csvw
import procasmt as pasmt


''' ========== User Input ========== '''

teacher = 'lewis_a_howell'
course = 'algebra2'
period = 'period2'
asmtdate = '04182018'
datadir = '/home/bryan/Desktop/edusoft_datatools/ClusterDatasets'


''' ========== File Information ========== '''

# get file names
asmt_fnames = pasmt.asmt_files(datadir, teacher, period, asmtdate)
ndir = len(asmt_fnames)
print(asmt_fnames)

# read files
all_asmt = []  # preallocate
for k in range(ndir):
    all_asmt.append(csvw.read_csv(asmt_fnames[k]))

# checker header
numhd = pasmt.ckheader(all_asmt)

''' ========== Assess Student Movement ========== '''

asmt_id, id_sets = pasmt.studmvt(all_asmt, numhd)
set_identer = id_sets.get('ID_enter_now')
set_idexit = id_sets.get('ID_exit_now')
print('Students that entered:', set_identer)
print('Students that exited:', set_idexit)
print('Students that permanently exited:',
      id_sets.get('ID_exit')-id_sets.get('ID_enter'))


''' ========== Reorganize Assessment ========== '''

allasmt_reorg = pasmt.reorg_asmt(
    all_asmt, numhd, asmt_id, id_sets.get('ID_all'))


''' ========== Merge Files ========== '''

cumdata = pasmt.combasmt(allasmt_reorg)


''' ========== Save Cumulative Assessment ========== '''

# writing file
tmp = asmt_fnames[ndir-1].rsplit('/', 1)
fnout = tmp[0]+'/cum'+tmp[1][1:]
# print(fnout)
csvw.write_csv(cumdata, fnout)
# if os.path.isfile(fnout)==False:
#     csvw.write_csv(cumdata,fnout)
