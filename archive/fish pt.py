import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('AI/archive/Fish.csv')

columns = ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"]

df = pd.DataFrame(data, columns=columns)

# Convert categorical species into numerical labels
df['Species'] = pd.Categorical(df['Species']).codes

# Prepare the data
X = df.drop("Weight", axis=1)  # Features
y = df["Weight"]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Select a Roach instance from the test set for prediction (you can adjust the index)
roach_instance = X_test.iloc[[0]]

# Transform features using the same polynomial transformation
roach_instance_poly = poly.transform(roach_instance)

# Predict weight for the Roach
roach_weight_prediction = model.predict(roach_instance_poly)

print("Predicted weight for Roach:", roach_weight_prediction)

import matplotlib.pyplot as plt

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Ideal')
plt.xlabel('Actual Weight')
plt.ylabel('Predicted Weight')
plt.title('Actual vs. Predicted Weight')
plt.legend()
plt.show()