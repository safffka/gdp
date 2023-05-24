import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Load the data
df = pd.read_csv('df.csv', index_col='date', parse_dates=True)
X = df.drop('gdp', axis=1)
y = df['gdp']

# Define range of components to test
num_components_range = range(1, X.shape[1] + 1)

# Initialize variables to store results
best_mae = float('inf')
best_rmse = float('inf')
best_num_components = None

# Define the number of folds for cross-validation
num_folds = 5

# Create TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=num_folds)

# Iterate over different numbers of components
for num_components in num_components_range:
    # Initialize variables to store fold results
    fold_mae = 0.0
    fold_rmse = 0.0

    # Perform cross-validation
    for train_index, test_index in tscv.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Perform feature scaling using StandardScaler within each fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Perform feature selection using Lasso within each fold
        lasso_model = Lasso(alpha=0.01)
        lasso_model.fit(X_train_scaled, y_train)
        X_train_lasso = X_train * lasso_model.coef_
        X_test_lasso = X_test * lasso_model.coef_

        # Perform PCA
        pca = PCA(n_components=num_components)
        X_train_pca = pca.fit_transform(X_train_lasso)
        X_test_pca = pca.transform(X_test_lasso)

        # Fit SARIMAX model with exogenous variables
        model = SARIMAX(
            y_train,
            exog=X_train_pca,
            order=(1, 0, 0),
            seasonal_order=(0, 1, 0, 4),
            enforce_invertibility=False
        )
        model_fit = model.fit()

        # Make predictions
        predictions = model_fit.predict(
            start=len(X_train_pca),
            end=len(X_train_pca) + len(X_test_pca) - 1,
            exog=X_test_pca
        )

        # Calculate MAE and RMSE for the fold
        fold_mae += mean_absolute_error(y_test, predictions)
        fold_rmse += np.sqrt(mean_squared_error(y_test, predictions))

    # Calculate average MAE and RMSE across folds
    fold_mae /= num_folds
    fold_rmse /= num_folds

    # Check if current model outperforms previous best model
    if fold_rmse < best_rmse:
        best_mae = fold_mae
        best_rmse = fold_rmse
        best_num_components = num_components
        predictions_best = predictions


# Print the best model's metrics
print(f"Best Number of Components: {best_num_components}")
print("Best Average MAE:", best_mae)
print("Best Average RMSE:", best_rmse)

#plot the best model's predictions against the actual values
plt.plot(predictions_best, label='Predicted')
plt.plot(y_test, label='Actual')
plt.legend()
plt.show()
