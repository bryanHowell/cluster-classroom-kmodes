'''
Created on Nov 1, 2017
@author Bryan Howell, Ph.D.

Description of csvwrap module.
This module is a set of wrapper functions that build on top of the csv module in python.
'''

# the wrapper functions use csv and os modules
import csv
import os


def read_csv(fname):
    '''read_csv reads a .csv file. 

    Parameters:
    fname, the file name to be read (.csv) 

    Returns:
    fdata, the data in the file that was read (.csv)
    '''
    f = open(fname, 'r')  # open file stream
    reader = csv.reader(f)  # collect contents of file stream
    fdata = []  # preallocate new data object
    for row in reader:  # for each stream of data up to a return line...
        fdata.append(row)  # append data row by row into a list
    f.close()  # close the file stream
    return fdata


def write_csv(data, fname):
    '''write_csv writes a .csv file.

    Parameters:
    data, the data to be written
    fname, the name of the file to be written

    Returns:
    integer: 1 if file written, 0 if file already exists
    '''
    if os.path.isfile(fname) == False:  # if the file doesn't already exist
        with open(fname, "w") as csv_file:  # open file stream for writing
            # define options for writing
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            writer.writerows(data)  # write all rows for the data
        print('file written successfully!')
        return 1
    else:
        print('File already exists.')
        return 0
