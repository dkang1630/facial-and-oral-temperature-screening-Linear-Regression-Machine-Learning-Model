import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

training_file_path = r'c:\Users\dkang\OneDrive\Documents\facial-and-oral-temperature-data-from-a-large-set-of-human-subject-volunteers-1.0.0\FLIR_groups1and2.csv'
testing_file_path = r"C:\Users\dkang\OneDrive\Documents\facial-and-oral-temperature-data-from-a-large-set-of-human-subject-volunteers-1.0.0\ICI_groups1and2.csv"

# Read the first three rows to use them as headers
# df = pd.read_csv(file_path, header=1, skiprows=[0, 2])
df = pd.read_csv(training_file_path, header=1)
df_test = pd.read_csv(testing_file_path, header=1)

# print(df.columns.tolist())
# print("\n")
# Rename all duplicate columns by appending an index
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

df.columns = rename_duplicates(df.columns)
df_test.columns = rename_duplicates(df_test.columns)
# print(df.columns.tolist())

# features_dict = df.to_dict(orient='list')


X = df[['T_FC', 'T_FC.1', 'T_FC.2', 'T_FC.3', 
         'T_FCmax', 'T_FCmax.1', 'T_FCmax.2', 'T_FCmax.3', 
         'T_CEmax', 'T_CEmax.1', 'T_CEmax.2', 'T_CEmax.3', 
         'T_max', 'T_max.1', 'T_max.2', 'T_max.3'
        #  'T_FEmax', 'T_FEmax.1', 'T_FEmax.2', 'T_FEmax.3'
        ]]
y = df['T_Mmax']

X_test = df_test[['T_FC', 'T_FC.1', 'T_FC.2', 'T_FC.3', 
         'T_FCmax', 'T_FCmax.1', 'T_FCmax.2', 'T_FCmax.3', 
         'T_CEmax', 'T_CEmax.1', 'T_CEmax.2', 'T_CEmax.3', 
         'T_max', 'T_max.1', 'T_max.2', 'T_max.3'
        #  'T_FEmax', 'T_FEmax.1', 'T_FEmax.2', 'T_FEmax.3'
         ]]
y_test = df_test['T_Mmax']


X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

X_test.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)

y = y[pd.to_numeric(y, errors='coerce').notnull()]
y_test = y_test[pd.to_numeric(y_test, errors='coerce').notnull()]

# Ensure X has the same length as y
X = X.iloc[y.index]
X_test = X_test.iloc[y_test.index]


# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

#Hyperparameter tuning of alpha with cross validation
# ridge = Ridge()
# param_grid = {'alpha': [1.0, 5.0, 6.0, 8.0]}
# grid_search = GridSearchCV(ridge, param_grid, scoring="neg_mean_squared_error", cv=5)
# grid_search.fit(X_train, y_train)
# best_alpha = grid_search.best_params_['alpha']
# print(f"Best alpha: {best_alpha}")

# Linear regression training
# regr = linear_model.LinearRegression()
# regr.fit(X_train, y_train)
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

