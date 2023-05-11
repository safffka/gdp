

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Lasso

export=np.load('export.npz')
export=export['export']
import_ru=np.load('import.npz')
import_ru=import_ru['import_ru']
moex=np.load('moex.npz')
moex=moex['moex']
gov_cons=np.load('gov_cons.npz')
gov_cons=gov_cons['gov_consumption']
inflation=np.load('inflation.npz')
inflation=inflation['inflation']
interest_rate=np.load('interest_rate.npz')
interest_rate=interest_rate['interest_rate']
investment=np.load('investment.npz')
investment=investment['investment']
unemployment=np.load('unemployment.npz')
unemployment=unemployment['unemployment']
money_supply=np.load('money_supply.npz')
money_supply=money_supply['money_supply']
oil_prices=np.load('oil_prices.npz')
oil_price=oil_prices['oil_price']
private_consumption=np.load('private_consumption.npz')
private_consumption=private_consumption['private_consumption']
rub_usd=np.load('rub_usd.npz')
rub_usd=rub_usd['rub_usd']
salary=np.load('salary.npz')
salary=salary['salary']

gdp =[2851.1,3101.7,3600.2,3655.2,3515.7,3971.6,4594.0,4945.9,4458.6,5077.9,5845.2,6228.1,5792.9,6368.1,7275.8,7480.3,6780.2,7767.5,8902.7,9797.0,8877.7,10238.3,11542.0,10618.,8334.6,9244.8,10411.3,10816.4,9995.8,10977.0,12086.5,13249.3,11954.2,13376.4,14732.9,15903.7,15182.8,
16436.0,17715.8,18768.9,16370.0,17507.9,19003.5,20104.3,17311.4,19044.2,20544.0,22130.5,18467.9,19751.0,21788.6,23079.8,18885.1,20452.2,22235.1,24043.6,20586.1,21917.6,23718.2,25621.2,22474.5,24969.8,27196.8,29220.6,24608.6,26628.6,28346.0,30025.1,24866.0,23775.4,27786.3,31230.5,27939.8,31885.1,35358.5]
#create dataframe
df=pd.DataFrame({'export':export,'import_ru':import_ru,'moex':moex,'gov_cons':gov_cons,'inflation':inflation,'interest_rate':interest_rate,'investment':investment,'unemployment':unemployment,'money_supply':money_supply,'oil_price':oil_price,'private_consumption':private_consumption,'rub_usd':rub_usd,'salary':salary})
#add gdp column

df['gdp']=gdp
#change dat type of columns to float
df=df.astype(float)
#add date column
df['date']=pd.date_range(start='2003-01-01', end='2021-09-30', freq='Q')
#set date as index
df.set_index('date', inplace=True)
#save df as csv
df.to_csv('df.csv')
#create correlation matrix
corr=df.corr()
#multiply columns by correlation matrix
#plot heatmap
plt.figure(figsize=(10,10))
plt.title('Correlation Matrix')
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.imshow(corr, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr)), corr.columns)
#save plt as png
plt.savefig('correlation_matrix.png')
tscv = TimeSeriesSplit(n_splits=5)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)


# Initialize lists to store the evaluation metrics
rf_mae_scores = []
rf_rmse_scores = []
gb_mae_scores = []
gb_rmse_scores = []
sarimax_mae_scores = []
sarimax_rmse_scores = []
order=(1,0,0)
seasonal_order=(1,0,0,4)



# Loop through the splits and train/test the models
for train_index, test_index in tscv.split(df):
    # Split the data into training and test sets
    X_train, X_test = df.iloc[train_index,:-1], df.iloc[test_index,:-1]
    y_train, y_test = df.iloc[train_index,-1], df.iloc[test_index,-1]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lasso_model = Lasso(alpha=0.1)

    # fit the model on the training data
    lasso_model.fit(X_train_scaled, y_train)

    #print the model coefficients
    X_train_scaled=X_train*lasso_model.coef_
    X_test_scaled=X_test*lasso_model.coef_



    # Fit the models
    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)
    model = SARIMAX(y_train, exog=X_train_scaled, order=(1, 0, 0), seasonal_order=(0, 1, 0, 4),
                    enforce_invertibility=False)
    results = model.fit()


    # Make predictions
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_gb = gb_model.predict(X_test_scaled)
    y_pred = results.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test_scaled)


    # Evaluate the models
    rf_mae_scores.append(mean_absolute_error(y_test, y_pred_rf))
    rf_rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    gb_mae_scores.append(mean_absolute_error(y_test, y_pred_gb))
    gb_rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_gb)))
    sarimax_mae_scores.append(mean_absolute_error(y_test, y_pred))
    sarimax_rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))


# Print the mean evaluation metrics
print('Random Forest MAE:', np.mean(rf_mae_scores))
print('Random Forest RMSE:', np.mean(rf_rmse_scores))
print('Gradient Boosting MAE:', np.mean(gb_mae_scores))
print('Gradient Boosting RMSE:', np.mean(gb_rmse_scores))
print('SARIMAX MAE:', np.mean(sarimax_mae_scores))
print('SARIMAX RMSE:', np.mean(sarimax_rmse_scores))

# Plot the predicted vs actual values
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_rf, label='Random Forest')
plt.plot(y_test.index, y_pred_gb, label='Gradient Boosting')
plt.plot(y_test.index, y_pred, label='SARIMAX')
plt.legend(loc='best')
plt.title('Predicted vs Actual GDP')
plt.ylabel('GDP')
plt.show()

#almon lag model






model = SARIMAX(y_train, exog=X_train_scaled, order=(1,0,0), seasonal_order=(0,1,0,4), enforce_invertibility=False)
results = model.fit()
y_pred = results.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test_scaled)
print('SARIMAX MAE:', mean_absolute_error(y_test, y_pred))
print('SARIMAX RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
#plot to compare SARIMAX and Gradient Boosting
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='SARIMAX')
plt.title('Actual vs Predicted GDP')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.legend()
plt.show()

#boost precision of SARIMAX model by using lasso to select features
#fit lasso model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)
#multiply columns by lasso coefficients
X_train_scaled=X_train*lasso_model.coef_
X_test_scaled=X_test*lasso_model.coef_
#fit SARIMAX model
model = SARIMAX(y_train, exog=X_train_scaled, order=(1,0,0), seasonal_order=(0,1,0,4), enforce_invertibility=False)
results = model.fit()
y_pred = results.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test_scaled)
print('SARIMAX MAE:', mean_absolute_error(y_test, y_pred))
print('SARIMAX RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))




