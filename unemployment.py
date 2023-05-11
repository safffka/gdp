import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


unemployment = pd.read_excel('unemployment.xls')
#delete first 9 rows
#delete first 3 rows

unemployment = unemployment.drop(unemployment.index[0:3])
#find the string whic contains 'Russia Federation'
unemployment = unemployment[unemployment['Data Source'].str.contains('Russian Federation')]
#keep 21 columns from 2003 to 2021
unemployment = unemployment.iloc[:, 47:67]
#transpose the data
unemployment = unemployment.transpose()
#change the index to year from 2003 to 2021
unemployment = unemployment.reset_index()
unemployment = unemployment.rename(columns={'index': 'year', 205: 'percent'})
#change year from 2003 to 2021
unemployment['year'] = pd.date_range(start='2003', end='2022', freq='Y')
#change value to float
unemployment['percent'] = unemployment['percent'].astype(float)
#change the size of inflation by copying the data for every quarter
unemployment = unemployment.reindex(np.repeat(unemployment.index.values, 4))
#change the index to date
unemployment = unemployment.reset_index()
unemployment = unemployment.rename(columns={'index': 'date'})
#change date from 2003-01-01 to 2021-12-31
unemployment['date'] = pd.date_range(start='2003-01-01', end='2021-12-31', freq='Q')
#delet last 2 rows
unemployment = unemployment.drop(unemployment.index[-1:])
#drop the column date
unemployment = unemployment.drop(['date'], axis=1)

#for each year, multiply the average gdp by the percent of investment
#plot the line graph

unemployment=unemployment['percent'].to_numpy()
np.savez('unemployment.npz', unemployment=unemployment)

