import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# **Step 1: Generate Random Dataset**
np.random.seed(42)
n_samples = 1000
n_features = 8

# Generate feature data (random values between 0 and 100)
data = np.random.rand(n_samples, n_features) * 100

# Generate coefficients for a linear relationship
coefficients = np.random.rand(n_features)

# Create target variable with some noise
target = data.dot(coefficients) + np.random.randn(n_samples) * 10  # Adding noise

# Create DataFrame
df = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(n_features)])
df['Target'] = target

# **Step 2: Split Features and Target**
X = df.drop(columns=['Target'])
y = df['Target']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Step 3: Initialize and Train Random Forest Model**
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# **Step 4: Make Predictions**
y_pred = rf.predict(X_test)

# **Step 5: Evaluate Model Performance**
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# **Step 6: Feature Importance Analysis**
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# **Step 7: Plot Feature Importances**
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Feature Importances in Random Forest Regression')
plt.show()

# **Step 8: Hyperparameter Tuning Using GridSearchCV**
param_grid = {
    'n_estimators': [50, 100],  # Reduced for faster execution
    'max_depth': [None, 10],  # Limited depth for efficiency
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# **Step 9: Evaluate Optimized Model**
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Optimized MSE:", mean_squared_error(y_test, y_pred_best))
print("Optimized R2 Score:", r2_score(y_test, y_pred_best))
