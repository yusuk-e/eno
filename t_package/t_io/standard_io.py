import os
import csv
import pickle
#import pandas as pd
import numpy as np

def check_dir(dir):
    #input:dir=path
    if not os.path.exists(dir):
        os.makedirs(dir)

def csv_read(file):
    #input:file=file name
    #output:D=np.array
    f = open(file)
    csvReader = csv.reader(f)
    D = []
    for row in csvReader:
        D.append(row)
    return D

def csv_write(file, D):
    #input:file=file name, D=np.array
    f = open(file,'w')
    csvWriter = csv.writer(f,lineterminator='\n')
    if np.ndim(D) == 1:
        csvWriter.writerow(D)
    elif np.ndim(D) == 2:
        for i in range(np.shape(D)[0]):
            line = D[i]
            csvWriter.writerow(line)
    f.close()

def pkl_read(file):
    #input:file=file name
    #output:D=dictionary
    f = open(file, 'rb')
    D = pickle.load(f)
    f.close()
    return D

def pkl_write(file, D):
    #input:file=file name, D=dictionary
    f = open(file,'wb')
    pickle.dump(D,f,protocol=4)
    f.close()

def df_read(file):
    #input:file=file name
    #output:D=dataframe
    #D = pd.read_pickle(file,compression='xz')
    return D

def df_write(file, D):
    #input:file=file name, D=dataframe
    D.to_pickle(file,compression='xz')
