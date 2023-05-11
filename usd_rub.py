import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


rub_usd = pd.read_csv('rub_usd.csv')
#create column average=open+close/2
rub_usd['average'] = (rub_usd['Open'] + rub_usd['Close']) / 2
#delete columns open,close,high,low,adj close,volume
rub_usd = rub_usd.drop(['Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
#create df average_quarterly = average of 4 months

rub_usd['Date'] = pd.to_datetime(rub_usd['Date'], format='%Y-%m-%d')
rub_usd = rub_usd.resample('Q', on='Date').mean()
#add 31.4, 30.6,29.8,29.0 to the beginning of the dataframe for quaters 2003-1, 2003-2, 2003-3, 2003-4
rub_usd = rub_usd.reset_index()
rub_usd = rub_usd.rename(columns={'Date': 'date', 'average': 'value'})
rub_usd = rub_usd._append({'date': '2003-01-01', 'value': 31.4}, ignore_index=True)
rub_usd = rub_usd._append({'date': '2003-04-01', 'value': 30.6}, ignore_index=True)
rub_usd = rub_usd._append({'date': '2003-07-01', 'value': 29.8}, ignore_index=True)
rub_usd = rub_usd._append({'date': '2003-10-01', 'value': 29.0}, ignore_index=True)
rub_usd['date'] = pd.to_datetime(rub_usd['date'], format='%Y-%m-%d')
rub_usd = rub_usd.sort_values(by='date')
#delete the last row
rub_usd = rub_usd[:-1]
#nparray of values
rub_usd = rub_usd['value'].to_numpy()
np.savez('rub_usd', rub_usd=rub_usd)