from math import log

import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, Series
import statsmodels.api as sm
import rpy2.robjects as R
from rpy2.robjects.packages import importr
from pandas import date_range
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# Load the data
df=pd.read_csv('df.csv', index_col='date', parse_dates=True)
gdp=df.gdp
w = gdp.resample('q').sum()
w_log = w.apply(lambda x: log(x))
#perform Dickey-Fuller test, null hypothesis is that the series is non-stationary

from statsmodels.tsa.stattools import adfuller
result = adfuller(w_log)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

#row is non-stationary

#perform adf test on first difference
w_log_diff = w_log.diff().dropna()
result = adfuller(w_log_diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

diff1lev_season = w_log_diff.diff(52).dropna()
#perform Xarki-Bera test, null hypothesis is that the series is normally distributed
from statsmodels.stats.stattools import jarque_bera
result = jarque_bera(diff1lev_season)
print('Jarque-Bera Statistic: %f' % result[0])
print('p-value: %f' % result[1])
d=1
#plot acf and pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(diff1lev_season, lags=10)
plot_pacf(diff1lev_season, lags=10)
plt.show()
p=1
q=1
