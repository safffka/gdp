import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


interest_rate = pd.read_excel('interest_rate.xls')
#delete first 9 rows
#delete first 3 rows

interest_rate = interest_rate.drop(interest_rate.index[0:3])
#find the string whic contains 'Russia Federation'
interest_rate = interest_rate[interest_rate['Data Source'].str.contains('Russian Federation')]
#keep 21 columns from 2003 to 2021
interest_rate = interest_rate.iloc[:, 47:67]
#transpose the data
interest_rate = interest_rate.transpose()
#change the index to year from 2003 to 2021
interest_rate = interest_rate.reset_index()
interest_rate = interest_rate.rename(columns={'index': 'year', 205: 'percent'})
#change year from 2003 to 2021
interest_rate['year'] = pd.date_range(start='2003', end='2022', freq='Y')
#change value to float
interest_rate['percent'] = interest_rate['percent'].astype(float)
#change the size of inflation by copying the data for every quarter
interest_rate = interest_rate.reindex(np.repeat(interest_rate.index.values, 4))
#change the index to date
interest_rate = interest_rate.reset_index()
interest_rate = interest_rate.rename(columns={'index': 'date'})
#change date from 2003-01-01 to 2021-12-31
interest_rate['date'] = pd.date_range(start='2003-01-01', end='2021-12-31', freq='Q')
#delet last 1 rows
interest_rate = interest_rate.drop(interest_rate.index[-1:])
#delete the column year
del interest_rate['date']


#for each year, multiply the average gdp by the percent of investment
#plot the line graph
plt.plot(interest_rate['year'], interest_rate['percent'])
plt.title('Interest Rate in Russia')
plt.xlabel('Date')
plt.ylabel('Percent')
plt.show()
print(interest_rate)

interest_rate=interest_rate['percent'].to_numpy()
np.savez('interest_rate.npz', interest_rate=interest_rate)
