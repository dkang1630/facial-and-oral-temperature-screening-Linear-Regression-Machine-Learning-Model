import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np 
from sklearn.linear_model import LinearRegression


def predict_lwr(query_point, X_train, y_train, tau):
    # Add a bias term to the training data
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]  # Add a column of ones

    # Add a bias term to the query point
    query_point_bias = np.insert(query_point, 0, 1)  # Insert 1 at the start for bias

    # Calculate distances from the query point to all training points
    distances = np.linalg.norm(X_train - query_point, axis=1)

    # Compute weights using Gaussian kernel
    weights = np.exp(- (distances ** 2) / (2 * tau ** 2))

    # Fit weighted linear regression
    weighted_model = LinearRegression()
    weighted_model.fit(X_train_bias, y_train, sample_weight=weights)

    # Make prediction
    return weighted_model.predict(query_point_bias.reshape(1, -1))[0]


def rename_duplicates(columns):
    seen = {}
    new_columns = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            new_columns.append(col)
    return new_columns

training_file_path = r"C:\Users\dkang\OneDrive\Documents\facial-and-oral-temperature-screening-Linear-Regression-Machine-Learning-Model\ICI_groups1and2.csv"
testing_file_path = r"C:\Users\dkang\OneDrive\Documents\facial-and-oral-temperature-screening-Linear-Regression-Machine-Learning-Model\FLIR_groups1and2.csv"

# Read the first 2 as headers
df = pd.read_csv(training_file_path, header = 1)
df_test = pd.read_csv(testing_file_path, header = 1)

# rename headers with duplicate names
df.columns = rename_duplicates(df.columns)
df_test.columns = rename_duplicates(df_test.columns)

# Attempt to select the desired columns
try:
    # Modify the tuples if your multi-level headers require them
    header = df[['T_FC', 'T_FC.1', 'T_FC.2', 'T_FC.3', 
         'T_FCmax', 'T_FCmax.1', 'T_FCmax.2', 'T_FCmax.3', 
         'T_CEmax', 'T_CEmax.1', 'T_CEmax.2', 'T_CEmax.3', 
         'T_max', 'T_max.1', 'T_max.2', 'T_max.3',
         'T_Mmax', 'T_Mmax.1','T_Mmax.2','T_Mmax.3']]
    print("Selected header from training DataFrame:")
    print(header.head())
except KeyError as e:
    print("KeyError: One or more of the specified columns are not found in the DataFrame.", e)


X = df[['T_FC', 'T_FC.1', 'T_FC.2', 'T_FC.3', 
         'T_FCmax', 'T_FCmax.1', 'T_FCmax.2', 'T_FCmax.3', 
         'T_CEmax', 'T_CEmax.1', 'T_CEmax.2', 'T_CEmax.3', 
         'T_max', 'T_max.1', 'T_max.2', 'T_max.3'
         ]]
y = df[['T_Mmax', 'T_Mmax.1','T_Mmax.2','T_Mmax.3']]

X_test = df_test[['T_FC', 'T_FC.1', 'T_FC.2', 'T_FC.3', 
         'T_FCmax', 'T_FCmax.1', 'T_FCmax.2', 'T_FCmax.3', 
         'T_CEmax', 'T_CEmax.1', 'T_CEmax.2', 'T_CEmax.3', 
         'T_max', 'T_max.1', 'T_max.2', 'T_max.3'
         ]]
y_test = df_test[['T_Mmax', 'T_Mmax.1','T_Mmax.2','T_Mmax.3']]

# Convert to numeric and handle NaN values, skipping invalid entries
X = X.apply(pd.to_numeric, errors='coerce')  # Convert and coerce invalid to NaN
y = y.apply(pd.to_numeric, errors='coerce')  

X.fillna(0, inplace=True)  # Fill NaNs with 0
y.fillna(0, inplace=True)  

X_test = X_test.apply(pd.to_numeric, errors='coerce')  # Convert and coerce invalid to NaN
y_test = y_test.apply(pd.to_numeric, errors='coerce')  

X_test.fillna(0, inplace=True)  # Fill NaNs with 0
y_test.fillna(0, inplace=True)  


# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

print(X_train.shape)
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

# L2 Regularization training 
ridge_model = Ridge(alpha=5.0)
ridge_model.fit(X_train, y_train)

# Linear regression prediction
# y_val_pred = regr.predict(X_val)
# y_test_pred = regr.predict(X_test)

y_val_pred = ridge_model.predict(X_val)
y_test_pred = ridge_model.predict(X_test)

# Evaluate the model
mae_val = mean_absolute_error(y_val, y_val_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the evaluation metrics
print("Mean Absolute Error for Validation (MAE):", mae_val)
print("Mean Squared Error for Validation(MSE):", mse_val)
print("R-squared for Validation:", r2_val)
print("\n")
print("Mean Absolute Error for testing (MAE):", mae_test)
print("Mean Squared Error for testing(MSE):", mse_test)
print("R-squared for testing:", r2_test)
# Print the coefficients and intercept
# print("Coefficients:", regr.coef_)
# print("Intercept:", regr.intercept_)



# LWR predictions for validation set
tau = 0.5
# LWR predictions for the validation set
lwr_val_pred = np.array([predict_lwr(X_val.iloc[i], X_train, y_train, tau) for i in range(X_val.shape[0])])

# Evaluate LWR performance on validation set
lwr_mae_val = mean_absolute_error(y_val, lwr_val_pred)
lwr_mse_val = mean_squared_error(y_val, lwr_val_pred)
lwr_r2_val = r2_score(y_val, lwr_val_pred)

print("\n")
print("LWR Validation Set Performance:")
print("MAE (Mean Absolute Error):", lwr_mae_val)
print("MSE (Mean Squared Error):", lwr_mse_val)
print("R-squared:", lwr_r2_val)

# LWR predictions for the test set
lwr_test_pred = np.array([predict_lwr(X_test.iloc[i], X_train, y_train, tau) for i in range(X_test.shape[0])])

# Evaluate LWR performance on the test set
lwr_mae_test = mean_absolute_error(y_test, lwr_test_pred)
lwr_mse_test = mean_squared_error(y_test, lwr_test_pred)
lwr_r2_test = r2_score(y_test, lwr_test_pred)

# Print LWR performance metrics for the test set
print("\nLWR Test Set Performance:")
print("MAE (Mean Absolute Error):", lwr_mae_test)
print("MSE (Mean Squared Error):", lwr_mse_test)
print("R-squared:", lwr_r2_test)




