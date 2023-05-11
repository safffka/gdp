import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



arr=[66.92,82.50,117.34,141.65,150.36,197.57,129.47,168.02,213.81,222.45,222.21,205.01,117.79,100,123.57,156.65,144.22,96.10,157.46]
arr=np.array(arr)
#duplicate the array 4 times
arr=np.repeat(arr,4)
#create dataframe from array with date from 2003-01-01 to 2021-12-31 and column name 'oil_price'
oil_price=pd.DataFrame({'date':pd.date_range(start='2003-01-01', end='2021-12-31', freq='Q'),'oil_price':arr})
#delete last 1 row
oil_price=oil_price.drop(oil_price.index[-1:])
#plot the line graph
plt.plot(oil_price['date'],oil_price['oil_price'])
plt.title('Oil Price in Russia')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
print(oil_price)
oil_price=oil_price['oil_price'].to_numpy()
np.savez('oil_prices.npz', oil_price=oil_price)
