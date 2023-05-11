import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


supply = pd.read_excel('money_supply.xlsx')
#delete the second column by index
supply = supply.drop(supply.columns[1], axis=1)
#rename the columns
supply = supply.rename(columns={'Unnamed: 0': 'Date', 'Money supply seasonally adjusted, billions of rubles': 'Money supply'})
#delete the first row
supply = supply.drop(supply.index[0])
#change the date to datetime
supply['Date'] = pd.to_datetime(supply['Date'], format='%Y-%m-%d')

#delete rows before Date < 2003 and Date > 2021
supply = supply[supply['Date'] >= '2003-01-01']
supply = supply[supply['Date'] <= '2021-12-31']
#change the money supply to float
supply['Money supply'] = supply['Money supply'].astype(float)
#get the average of every quarter
supply = supply.resample('Q', on='Date').mean()
supply = supply.drop(supply.index[0])
#linear plot
plt.plot(supply['Money supply'])
plt.title('Money supply in Russia')
plt.xlabel('Date')
plt.ylabel('Billions of rubles')
plt.show()
money_supply=supply['Money supply'].to_numpy()
np.savez('money_supply.npz', money_supply=money_supply)



