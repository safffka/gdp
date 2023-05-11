import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


gov_consumption = pd.read_excel('gov_cons.xls')
#delete first 9 rows
gov_consumption = gov_consumption.drop(gov_consumption.index[0:9])
#make first row the header
gov_consumption.columns = gov_consumption.iloc[0]
gov_consumption = gov_consumption.drop(gov_consumption.index[0])
#change observation_date to 'yyyy-mm-dd' format
gov_consumption['observation_date'] = pd.to_datetime(gov_consumption['observation_date'], format='%Y-%m-%d')
#set observation_date as index
gov_consumption = gov_consumption.set_index('observation_date')
#sepate the data into two columns 'date' and 'value'
gov_consumption = gov_consumption.reset_index()
#rename columns
gov_consumption = gov_consumption.rename(columns={'observation_date': 'date', 'RUSGFCEQDSMEI': 'value'})
#change value to float
gov_consumption['value'] = gov_consumption['value'].astype(float)
#line plot
plt.plot(gov_consumption['date'], gov_consumption['value'])
plt.title('Government Consumption in Russia')
plt.xlabel('Date')
plt.ylabel('Rubles')
plt.show()
print(gov_consumption)

gov_consumption=gov_consumption['value'].to_numpy()
np.savez('gov_cons.npz', gov_consumption=gov_consumption)





