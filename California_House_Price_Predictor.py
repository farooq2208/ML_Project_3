""" End-to-End XGBoost Pipeline for California House Price Prediction"""

# Importing Libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

# Loading Dataset
df = pd.read_csv('dataset.csv')

# Detcting Missing Values
df.isna().sum()      #After printing,  In "total_bedrooms" column, found 207 missing values

# Fill with median of that ocean_proximity group
df['total_bedrooms'] = df.groupby('ocean_proximity')['total_bedrooms'].transform(
    lambda x: x.fillna(x.median())
 )

# Removing bad labels
df = df[df['median_house_value'] < 500001]

# Feature Engineering
df['rooms_per_household']       = df['total_rooms'] / df['households']
df['bedrooms_per_room']         = df['total_bedrooms'] / df['total_rooms']
df['population_per_household']  = df['population'] / df['households']

# Dropping redundant columns
df = df.drop(['total_rooms','total_bedrooms','population','households'], axis=1)

# Separating Data into features (X) and target (y)
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

# Splitting features and target in train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Selecting Categorical Columns
cat_features = X.select_dtypes(exclude='number').columns

# Creatng Categorical Pipeline to transform Categorical Columns
cat_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop = 'first', handle_unknown='ignore'))
    ])

# Columntransformer
preprocessor = ColumnTransformer(transformers=[
    ('encoder', cat_pipeline, cat_features)
], remainder='passthrough')  # This will keep include numeric features as well

# Final Pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor())
])
# Developing Param_grid
param_grid = [{
    'model': [XGBRegressor(random_state=42)],
    'model__n_estimators': [500, 800],
    'model__learning_rate': [0.05, 0.08],
    'model__max_depth': [6, 8],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8]
}]

# GridSearchCV
grid = GridSearchCV(final_pipeline, param_grid, cv=5, n_jobs=-1, scoring='r2', verbose=1)

# Fitting Grid
grid.fit(X_train, y_train)

# Determining Best Estimator
print(grid.best_params_)
model = grid.best_estimator_

# Predictions
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Metrics Evaluation
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)

# Calculating Residuals
residuals = y_test - y_pred_test

# Finding target Range
y_range = y_test.max() - y_test.min()

#  Finding error relative to full price range.
rmse_over_range = rmse / y_range * 100

# Displaying Results
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Testing: {r2_test:.4f}")
print(f"R² Training: {r2_train:.4f}")
print(f"Max Residual: {max(abs(residuals)):.2f}")
print(f"STD of Residuals: {residuals.std():.2f}")
print(f"y_range : {y_range:.2f}")
print(f"10. RMSE / y_range: {rmse_over_range:.2f}%")

# CV best score
print(f"Best CV R2: {grid.best_score_:.4f}")

# Saving Model
joblib.dump(model, 'California_House_Price_Prediction.pkl')