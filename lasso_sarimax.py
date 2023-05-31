import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

# Load the data
df = pd.read_csv('df.csv', index_col='date', parse_dates=True)
X = df.drop('gdp', axis=1)
y = df['gdp']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the alpha values to be tested
alpha_values = np.arange(0.01, 10.01, 0.01)

# Define the range of splits for time series cross-validation
n_splits_range = range(2, 11)

# Initialize variables to store the best alpha, n_splits, and AIC
best_alpha = None
best_n_splits = None
best_aic = np.inf

# Perform time series cross-validation with varying n_splits and alpha
for n_splits in n_splits_range:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for alpha in alpha_values:
        aic_sum = 0
        for train_index, test_index in tscv.split(X_scaled):
            # Split the data into training and testing sets
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the Lasso regression model
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_train, y_train)

            # Predict the test set
            y_pred_lasso = lasso.predict(X_test)

            # Calculate the log-likelihood
            residuals = y_test - y_pred_lasso
            sigma_squared = np.mean(residuals ** 2)
            log_likelihood = -0.5 * len(y_test) * np.log(2 * np.pi * sigma_squared) - 0.5 * np.sum(residuals ** 2) / sigma_squared

            # Calculate the AIC for the current fold
            n = len(y_test)
            k = X_test.shape[1] + 1  # Number of predictors + intercept
            aic = -2 * log_likelihood + 2 * n * k

            # Sum up the AIC across folds
            aic_sum += aic

        # Calculate the average AIC across folds
        avg_aic = aic_sum / n_splits

        # Check if the current alpha and n_splits have a lower AIC
        if avg_aic < best_aic:
            best_aic = avg_aic
            best_alpha = alpha
            best_n_splits = n_splits

# Print the best alpha, n_splits, and the corresponding AIC
print("Best alpha:", best_alpha)
print("Best n_splits:", best_n_splits)
print("Best AIC:", best_aic)

# Fit the Lasso regression model with the best alpha and n_splits
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_scaled, y)

# Predict the entire dataset using the Lasso model
y_pred_lasso = lasso.predict(X_scaled)

# Calculate the residuals
residuals = y - y_pred_lasso

# Define the specifications for seasonal unit roots and trends
sea_d = [0, 1]  # different specifications for seasonal unit root
trend = [None, 'c', 'ct']  # different specifications for including trend or not
models = list(product(sea_d, trend))

# Initialize variables to store the best ARIMA model and its BIC
best_model = None
best_bic = np.inf

# Evaluate different specifications of seasonal unit roots and trends
for jj, kk in enumerate(models):
    # Fit the ARIMA model
    model = pm.auto_arima(residuals, D=kk[0], stepwise=False, trend=kk[1], seasonal=True, m=12, test='adf',
                          with_intercept='auto', ic='aic', seasonal_test='ocsb', error_action='ignore')
    # Calculate the BIC
    bic = model.bic()

    # Check if the current model has a lower BIC
    if bic < best_bic:
        best_bic = bic
        best_model = model

# Print the best ARIMA model parameters and its BIC
print("Best ARIMA Model Parameters:")
print(best_model.order)
print("Best BIC:", best_bic)

# Predict using the best ARIMA model
y_pred_arima = best_model.predict(len(y))

# Combine the Lasso and ARIMA predictions
y_pred_combined = y_pred_lasso + y_pred_arima

# Calculate the log-likelihood
sigma_squared_combined = np.mean(residuals ** 2)
log_likelihood_combined = -0.5 * len(y) * np.log(2 * np.pi * sigma_squared_combined) - 0.5 * np.sum(residuals ** 2) / sigma_squared_combined

# Calculate the AIC for the combined model
n_combined = len(y)
k_combined = X_scaled.shape[1] + best_model.order[0] + best_model.order[1] + best_model.order[2] + 1  # Number of predictors + AR + I + MA + intercept
aic_combined = -2 * log_likelihood_combined + 2 * n_combined * k_combined

# Print the AIC for the combined model
print("AIC for the combined model:", aic_combined)


