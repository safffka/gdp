import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import gdp_avg from gdp.py
arr=[2851.1,3101.7,3600.2,3655.2,3515.7,3971.6,4594.0,4945.9,4458.6,5077.9,5845.2,6228.1,5792.9,6368.1,7275.8,7480.3,6780.2,7767.5,8902.7,9797.0,8877.7,10238.3,11542.0,10618.,8334.6,9244.8,10411.3,10816.4,9995.8,10977.0,12086.5,13249.3,11954.2,13376.4,14732.9,15903.7,15182.8,
16436.0,17715.8,18768.9,16370.0,17507.9,19003.5,20104.3,17311.4,19044.2,20544.0,22130.5,18467.9,19751.0,21788.6,23079.8,18885.1,20452.2,22235.1,24043.6,20586.1,21917.6,23718.2,25621.2,22474.5,24969.8,27196.8,29220.6,24608.6,26628.6,28346.0,30025.1,24866.0,23775.4,27786.3,31230.5,27939.8,31885.1,35358.5,36000]


investment= pd.read_excel('investments.xls')
#delete first 3 rows
investment = investment.drop(investment.index[0:3])
#find the string whic contains 'Russia Federation'
investment = investment[investment['Data Source'].str.contains('Russian Federation')]
#keep 21 columns from 2003 to 2021
investment = investment.iloc[:, 47:67]
#transpose the data
investment = investment.transpose()
#change the index to year from 2003 to 2021
investment = investment.reindex(np.repeat(investment.index.values, 4))
investment = investment.reset_index()
investment = investment.rename(columns={'index': 'year', 205: 'percent'})
#change year from 2003 to 2021
investment['year'] = pd.date_range(start='2003', end='2022', freq='Q')
#change value to float
investment['percent'] = investment['percent'].astype(float)
#for each year, multiply the average gdp by the percent of investment
investment['value'] = investment['percent'] * arr


investment=investment['value'].to_numpy()
#delete the last value
investment = np.delete(investment, -1)
np.savez('investment.npz', investment=investment)


