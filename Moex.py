import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


moex=pd.read_csv('MOEX .csv')
#reverse the data
moex=moex.iloc[::-1]
#drop 'Open','High','Low','Vol.','Change %'
moex=moex.drop(['Open','High','Low','Vol.','Change %'],axis=1)
#date as index
moex=moex.set_index('Date')
#resample to quarterly
moex.index=pd.to_datetime(moex.index,format='mixed')
#price as float
moex['Price']=moex['Price'].str.replace(',','').astype(float)


moex=moex.resample('Q').mean()
#delete the las row
moex=moex[:-1]
moex=moex['Price'].to_numpy()
np.savez('moex.npz',moex=moex)


