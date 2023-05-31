
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('df.csv', index_col='date', parse_dates=True)
X = df.drop('gdp', axis=1)
y = df['gdp']




model = auto_arima(y, seasonal=True, m=12)
order = model.get_params()['order']
seasonal_order = model.get_params()['seasonal_order']

sarimax_model = SARIMAX(y, order=order, seasonal_order=seasonal_order, loss='hubert', trend='c',  enforce_stationarity=False)

result = sarimax_model.fit(seasonal_order='Q',trend='c')
predicted_values = result.predict(start=X.index[0], end=X.index[-1])
mse = mean_squared_error(y, predicted_values)
aic = result.aic
print("Mean Squared Error:", mse)
print("AIC:", aic)
print(result.summary())




