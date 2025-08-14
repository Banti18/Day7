import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Step 1: Load and prepare the dataset ---
# Load the dataset from the uploaded CSV file.
file_path = "breast-cancer.csv"
df = pd.read_csv(file_path)

# Drop the 'id' column as it's not a feature for classification.
df.drop('id', axis=1, inplace=True)

# Separate features (X) and target (y).
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# The diagnosis column has 'M' (Malignant) and 'B' (Benign). We need to encode it.
le = LabelEncoder()
y = le.fit_transform(y)  # 'B' -> 0, 'M' -> 1

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features for better SVM performance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 2: Train an SVM with linear and RBF kernel ---

print("--- Training Linear Kernel SVM ---")
# Create an SVM classifier with a linear kernel.
linear_svm = SVC(kernel='linear', random_state=42)
# Train the model.
linear_svm.fit(X_train_scaled, y_train)
# Predict on the test data.
linear_pred = linear_svm.predict(X_test_scaled)
# Evaluate the model.
print(f"Accuracy: {accuracy_score(y_test, linear_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, linear_pred))

print("\n--- Training RBF Kernel SVM ---")
# Create an SVM classifier with an RBF (non-linear) kernel.
rbf_svm = SVC(kernel='rbf', random_state=42)
# Train the model.
rbf_svm.fit(X_train_scaled, y_train)
# Predict on the test data.
rbf_pred = rbf_svm.predict(X_test_scaled)
# Evaluate the model.
print(f"Accuracy: {accuracy_score(y_test, rbf_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, rbf_pred))


# --- Step 3: Visualize decision boundary using 2D data ---
# For visualization, we'll use only two features: 'radius_mean' and 'texture_mean'.
# This allows us to plot the decision boundary on a 2D plane.
X_2d = X[['radius_mean', 'texture_mean']]
X_2d_train, X_2d_test, y_2d_train, y_2d_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)

# Standardize the 2D data
scaler_2d = StandardScaler()
X_2d_train_scaled = scaler_2d.fit_transform(X_2d_train)
X_2d_test_scaled = scaler_2d.transform(X_2d_test)

# Train an RBF SVM on the 2D data
svm_2d = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_2d.fit(X_2d_train_scaled, y_2d_train)

# Create a meshgrid to plot the decision boundary
x_min, x_max = X_2d_train_scaled[:, 0].min() - 0.5, X_2d_train_scaled[:, 0].max() + 0.5
y_min, y_max = X_2d_train_scaled[:, 1].min() - 0.5, X_2d_train_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Plot the decision boundary
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

# Plot the training points
sns.scatterplot(x=X_2d_train_scaled[:, 0], y=X_2d_train_scaled[:, 1], hue=y_2d_train, palette='coolwarm', s=80, edgecolors='k')
plt.title('SVM Decision Boundary (RBF Kernel)')
plt.xlabel('Scaled radius_mean')
plt.ylabel('Scaled texture_mean')
plt.show()


# --- Step 4 & 5: Tune hyperparameters with cross-validation ---
print("\n--- Tuning Hyperparameters with GridSearchCV ---")
# Define the parameter grid to search over.
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Create a GridSearchCV object. It will perform k-fold cross-validation
# for each combination of C and gamma.
grid_search = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=2)

# Fit the grid search to the training data.
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and the best score found.
print("\nBest Parameters found:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# Use the best estimator to make predictions on the test set.
best_svm_pred = grid_search.predict(X_test_scaled)
print(f"\nAccuracy of Best SVM Model: {accuracy_score(y_test, best_svm_pred):.4f}")
print("Classification Report of Best Model:\n", classification_report(y_test, best_svm_pred))
