import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate random dataset
np.random.seed(42)
n_samples = 1000
n_features = 8

data = np.random.rand(n_samples, n_features) * 100
target = np.random.rand(n_samples) * 100

# Create DataFrame
df = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(n_features)])
df['Target'] = target

# Splitting features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Feature Importance Analysis
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Feature Importances in Random Forest Regression')
plt.show()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and evaluation
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Optimized MSE:", mean_squared_error(y_test, y_pred_best))
print("Optimized R2 Score:", r2_score(y_test, y_pred_best))