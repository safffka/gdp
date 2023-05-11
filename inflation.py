import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


inflation = pd.read_excel('inflation.xls')
#delete first 9 rows
#delete first 3 rows

inflation = inflation.drop(inflation.index[0:3])
#find the string whic contains 'Russia Federation'
inflation = inflation[inflation['Data Source'].str.contains('Russian Federation')]
#keep 21 columns from 2003 to 2021
inflation = inflation.iloc[:, 47:67]
#transpose the data
inflation = inflation.transpose()
#change the index to year from 2003 to 2021
inflation = inflation.reset_index()
inflation = inflation.rename(columns={'index': 'year', 205: 'percent'})
#change year from 2003 to 2021
inflation['year'] = pd.date_range(start='2003', end='2022', freq='Y')
#change value to float
inflation['percent'] = inflation['percent'].astype(float)
#change the size of inflation by copying the data for every quarter
inflation = inflation.reindex(np.repeat(inflation.index.values, 4))
#change the index to date
inflation = inflation.reset_index()
inflation = inflation.rename(columns={'index': 'date'})
#change date from 2003-01-01 to 2021-12-31
inflation['date'] = pd.date_range(start='2003-01-01', end='2021-12-31', freq='Q')
#delet last 2 rows
inflation = inflation.drop(inflation.index[-1:])


#for each year, multiply the average gdp by the percent of investment



inflation=inflation['percent'].to_numpy()
np.savez('inflation.npz', inflation=inflation)
