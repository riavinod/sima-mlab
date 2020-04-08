import pandas as pd
import numpy as np
import sys


df = pd.read_csv(sys.argv[1], delimiter=',')

def clean_col(column):
    lst = []
    #print(column)
    for i in range(column.shape[0]):
        elem = column[i].split()[0]
        #print(elem)
        #x = int(elem[0])
        if(elem != "utilization.gpu"):
            lst.append(int(float(elem)))
    return lst

def get_col_max(df):
    for i in range(df.columns.shape[0]):
        print(df.columns[i], df[df.columns[i]].max())

def get_col_mean(df):
    for i in range(df.columns.shape[0]):
        print(df.columns[i], df[df.columns[i]].mean())



col_names = df.columns
for col_name in col_names:
    df[col_name] = clean_col(df[col_name])

print("Maximum values by metric:")
get_col_max(df)
print('\n')
print("Mean values by metric:")
get_col_mean(df)
