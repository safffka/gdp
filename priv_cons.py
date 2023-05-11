import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


private_consumption = pd.read_excel('priv_cons.xls')
#delete first 9 rows
private_consumption = private_consumption.drop(private_consumption.index[0:9])
#make first row the header
private_consumption.columns = private_consumption.iloc[0]
private_consumption = private_consumption.drop(private_consumption.index[0])
#change observation_date to 'yyyy-mm-dd' format
private_consumption['observation_date'] = pd.to_datetime(private_consumption['observation_date'], format='%Y-%m-%d')
#set observation_date as index
private_consumption = private_consumption.set_index('observation_date')
#sepate the data into two columns 'date' and 'value'
private_consumption = private_consumption.reset_index()
#rename columns
private_consumption = private_consumption.rename(columns={'observation_date': 'date', 'RUSPFCEQDSMEI': 'value'})
#change value to float
private_consumption['value'] = private_consumption['value'].astype(float)
#line plot
plt.plot(private_consumption['date'], private_consumption['value'])
plt.title('Private Consumption in Russia')
plt.xlabel('Date')
plt.ylabel('Rubles')
plt.show()
private_consumption=private_consumption['value'].to_numpy()
np.savez('private_consumption.npz', private_consumption=private_consumption)



