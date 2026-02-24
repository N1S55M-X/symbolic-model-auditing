# Install PySR if not already installed
# !pip install pysr

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pysr import PySRRegressor

# Step 1: Load the Iris dataset
data = load_iris()
X = data.data  # Features (sepal length, sepal width, petal length, petal width)
y = data.target  # Target (species)

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define and train the PySR model
model = PySRRegressor(
    niterations=50,  # Number of iterations
    binary_operators=["+", "-", "*", "/"],  # Binary operators
    unary_operators=["square", "sqrt", "abs"],  # Unary operators
    populations=20,  # Number of populations
    population_size=50,  # Size of each population
    maxsize=20,  # Maximum size of an expression
    parsimony=0.01,  # Encourages simpler expressions
    procs=4,  # Number of CPU cores to use
    random_state=42,  # Random seed
)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the model on the test set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (Test Set): {mae:.4f}")

# Step 6: Validate on out-of-sample data
X_out_of_sample = X_train[:10]  # Use a small subset for out-of-sample validation
y_true_out_of_sample = y_train[:10]
y_pred_out_of_sample = model.predict(X_out_of_sample)

# Calculate MAE for out-of-sample data
mae_out_of_sample = mean_absolute_error(y_true_out_of_sample, y_pred_out_of_sample)
print(f"Mean Absolute Error (Out-of-Sample): {mae_out_of_sample:.4f}")

# Step 7: Display the best equation
print("\n🔹 Best Equation:")
print(model.sympy())

# Step 8: Plot the results (for the first feature vs. target)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.scatter(X_test[:, 0], y_test, color='blue', label="True Values", alpha=0.6)
plt.scatter(X_test[:, 0], y_pred, color='red', label="Predicted Values", alpha=0.6)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Species")
plt.title("Comparison: True vs. Predicted Values")
plt.legend()
plt.show()
