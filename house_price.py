from helpers import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df = pd.concat([df_train, df_test])
df.info()
df.describe()

# Exploratory Data Analysis
# Number of unique and NaN values
df.nunique()
df.isna().sum()
df.head()

# Classify column types and explore summary with respect to target variable
cat_cols, num_cols, cat_but_car = classify_columns_types(df, 26, 37)
numeric_column_summary(df, num_cols, "SalePrice")
categorical_column_summary(df, cat_cols, "SalePrice")

# Data Preparation & Feature Engineering
# Drop ID column
df = df.drop(columns="Id")

# Filling the gaps in the columns with the expression "No"
no_cols = ["Alley", "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond", "PoolQC", "Fence", "MiscFeature"]

for col in no_cols:
    df[col].fillna("No")

# Eliminate variable that include greater than 99% ratio of number of unique value
df = drop_highly_unique_columns(df, cat_cols)
cat_cols, num_cols, cat_but_car = classify_columns_types(df, 26, 37)

# Drop columns that have more than 90% NaN values
df = drop_high_nan_columns_rows(df)

# Reassignment for some variables
quality_mapping_with_no = {
    'No': 0,
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

quality_mapping_without_no = {
    'Po': 0,
    'Fa': 1,
    'TA': 2,
    'Gd': 3,
    'Ex': 4
}
quality_mapping_variables = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'HeatingQC', 'KitchenQual', 'FireplaceQu',
        'GarageQual', 'GarageCond']

for column in quality_mapping_variables:
    if 'No' in df[column].unique():
        df[column] = df[column].map(quality_mapping_with_no)
    else:
        df[column] = df[column].map(quality_mapping_without_no)

# Create variable 'other' for variables that include less than or equal to 5% ratio of number of unique value
cat_cols, num_cols, cat_but_car = classify_columns_types(df, 26, 37)
df = modify_low_unique_columns(df, cat_cols, 5)

# Determine outlier and replace with thresholds
plot_quartiles_outlier(df, num_cols)
for col in num_cols:
    if col != 'SalePrice':
        replace_outliers_with_thresholds(df, col, 0.05, 0.95)

# One-hot encoding
df = one_hot_encoder(df, cat_cols, True)

# Select columns to impute (excluding the target variable)
columns_to_impute = df.columns[df.columns != 'SalePrice']

# Create an instance of KNNImputer
imputer = KNNImputer(n_neighbors=5)

# Impute NaN values in selected columns
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
df.head()

# Standardization
num_cols = [col for col in num_cols if 'SalePrice' not in col]
standardization(df, num_cols, RobustScaler())

# Modeling

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

# Define X and y variables
y = train_df["SalePrice"]
X = train_df.drop("SalePrice", axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

# Linear Regression
linreg = LinearRegression().fit(X_train, y_train)
evaluate_model(linreg, X_train, y_train, X_test, y_test, "Linear Regression")

# Linear Regression Train RMSE: 26430.10
# Linear Regression Train R^2: 0.889
# Linear Regression Test RMSE: 30501.39
# Linear Regression Test R^2: 0.855

# Decision Tree Regressor
cart = DecisionTreeRegressor().fit(X_train, y_train)
evaluate_model(cart, X_train, y_train, X_test, y_test, "Decision Tree Regressor")
plot_importance(cart, X_test)
cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30) }
tune_model(cart, cart_params,  X_train, y_train, X_test, y_test, "Decision Tree Regressor")

# Decision Tree Regressor Train RMSE: 0.00
# Decision Tree Regressor Train R^2: 1.000
# Decision Tree Regressor Test RMSE: 50920.06
# Decision Tree Regressor Test R^2: 0.595

# Best Decision Tree Regressor Parameters: {'max_depth': 17, 'min_samples_split': 26}
# Tuned Decision Tree Regressor Train RMSE: 23637.92
# Tuned Decision Tree Regressor Train R^2: 0.911
# Tuned Decision Tree Regressor Test RMSE: 38924.96
# Tuned Decision Tree Regressor Test R^2: 0.763

# Random Forest
rf_model = RandomForestRegressor().fit(X_train, y_train)
evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
plot_importance(rf_model, X_test)
rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [5,15, 20],
             "n_estimators": [200, 300,500]}
tune_model(rf_model, rf_params,  X_train, y_train, X_test, y_test, "Random Forest")

# Random Forest Train RMSE: 11791.39
# Random Forest Train R^2: 0.978
# Random Forest Test RMSE: 28629.25
# Random Forest Test R^2: 0.872

# Best Random Forest Parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 200}
# Tuned Random Forest Train RMSE: 15558.19
# Tuned Random Forest Train R^2: 0.961
# Tuned Random Forest Test RMSE: 28904.01
# Tuned Random Forest Test R^2: 0.869

# XGBoost
xgboost = XGBRegressor().fit(X_train, y_train)
evaluate_model(xgboost, X_train, y_train, X_test, y_test, "XGBoost")
plot_importance(xgboost, X_test)
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100,200,500],
                  "colsample_bytree": [0.5, 1]}
tune_model(xgboost, xgboost_params,  X_train, y_train, X_test, y_test, "XGBoost")

# LightGBM - Light Gradient Boosting Model
lgb_model = LGBMRegressor().fit(X_train, y_train)
evaluate_model(lgb_model, X_train, y_train, X_test, y_test, "Light Gradient Boosting Model")
plot_importance(lgb_model, X_test)
lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500,1000],
                   "colsample_bytree": [0.7, 1]}

tune_model(lgb_model, lightgbm_params,  X_train, y_train, X_test, y_test, "Light Gradient Boosting Model")

# [LightGBM] [Info] Total Bins 3074
# [LightGBM] [Info] Number of data points in the train set: 1168, number of used features: 147
# [LightGBM] [Info] Start training from score 180664.512842
# Light Gradient Boosting Model Train RMSE: 11435.85
# Light Gradient Boosting Model Train R^2: 0.979
# Light Gradient Boosting Model Test RMSE: 26430.79
# Light Gradient Boosting Model Test R^2: 0.891

# Best Light Gradient Boosting Model Parameters: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 1000}
# Tuned Light Gradient Boosting Model Train RMSE: 11601.85
# Tuned Light Gradient Boosting Model Train R^2: 0.979
# Tuned Light Gradient Boosting Model Test RMSE: 25503.00
# Tuned Light Gradient Boosting Model Test R^2: 0.898