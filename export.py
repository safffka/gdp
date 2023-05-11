import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_excel('export.xls')
#transpose the dataframe
df=df.T

arr=df.to_numpy()
#arr1=average for every quarter
#append copy of last row
arr=np.append(arr,arr[-1])
arr=np.append(arr,arr[-1])
print(len(arr))
new_len=int(len(arr)/3)
array1=np.zeros(new_len)
for i in range(new_len):
    chunk = arr[i*3 : i*3+3]
    array1[i] = sum(chunk) / len(chunk)
#delete the last
array1=array1[:-1]
export=array1
np.savez('export.npz', export=export)

