'''
テキストファイルの書き込み
o_f = open(o_file, 'w')
o_f.writelines(line)
o_f.close()

np.save(file, D):numpy形式のデータを保存
    #input:file=file name(.npy), D=np.array

np.load(file):numpy形式のデータの読み込み
    #input:file=file name(.npy)

check_dir(dir):ディレクトリがなければ作る
    #input:dir=path

csv_read(file):csvファイルの読み込み
    #input:file=file name
    #output:D=np.array

csv_write(file, D):csvファイルの書き込み
    #input:file=file name, D=np.array

pkl_read(file):{}やDataframeで定義されたデータの読み込み(pickle)
    #input:file=file name
    #output:D=dictionary / dataframe

pkl_write(file, D):{}やDataframeで定義されたデータの書き込み(pickle)
    #input:file=file name, D=dictionary / dataframe

def df_read(file):dataframeで定義されたデータの読み込み(pickle)
    #input:file=file name
    #output:D=dataframe

def df_write(file, D):dataframeで定義されたデータの書き込み(pickle)
    #input:file=file name, D=dataframe
'''
import os
import csv
import pickle
#import pandas as pd
import numpy as np

def check_dir(dir):
    #ディレクトリがなければ作る
    #input:dir=path
    if not os.path.exists(dir):
        os.makedirs(dir)

def csv_read(file):
    #csvファイルの読み込み
    #input:file=file name
    #output:D=np.array
    f = open(file)
    csvReader = csv.reader(f)
    D = []
    for row in csvReader:
        D.append(row)
    return D

def csv_write(file, D):
    #csvファイルの書き込み
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
    #{}で定義されたデータの読み込み(pickle)
    #input:file=file name
    #output:D=dictionary
    f = open(file, 'rb')
    D = pickle.load(f)
    f.close()
    return D

def pkl_write(file, D):
    #{}で定義されたデータの書き込み(pickle)
    #input:file=file name, D=dictionary
    f = open(file,'wb')
    pickle.dump(D,f,protocol=4)
    f.close()

def df_read(file):
    #dataframeで定義されたデータの読み込み(pickle)
    #input:file=file name
    #output:D=dataframe
    #D = pd.read_pickle(file,compression='xz')
    return D

def df_write(file, D):
    #dataframeで定義されたデータの書き込み(pickle)
    #input:file=file name, D=dataframe
    D.to_pickle(file,compression='xz')
