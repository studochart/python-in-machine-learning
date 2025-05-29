# Machine Learning Practice Questions - Solutions
# These are solutions to the practice questions from the Machine Learning tutorial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, classification_report

# Set a random seed for reproducibility
np.random.seed(42)

# Question 1: Load the iris dataset and split it into training and testing sets (80% training, 20% testing)
print("Solution to Question 1: Split Iris Dataset")
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Question 2: Train a Decision Tree classifier on the iris dataset and report its accuracy
print("\nSolution to Question 2: Decision Tree Classifier")
# Create and train a decision tree classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Question 3: Create a scatter plot of the first two features of the iris dataset, coloring points by their species
print("\nSolution to Question 3: Scatter Plot of Iris Features")
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=70, alpha=0.7)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris Dataset: Sepal Length vs Sepal Width')
plt.colorbar(scatter, label='Species')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Question 4: Implement a K-means clustering algorithm on the iris dataset with 3 clusters and visualize the results
print("\nSolution to Question 4: K-means Clustering")
# Create and fit the K-means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster assignments
clusters = kmeans.labels_

# Visualize the clusters (using first two features)
plt.figure(figsize=(12, 5))

# Plot true classes
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('True Classes')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Plot K-means clusters
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
plt.title('K-means Clusters')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()

# Calculate adjusted rand index
ari = metrics.adjusted_rand_score(y, clusters)
print(f"Adjusted Rand Index: {ari:.4f}")

# Question 5: Train a Linear Regression model on the Boston Housing dataset to predict house prices
print("\nSolution to Question 5: Linear Regression on Boston Housing")
# Load the Boston Housing dataset
boston = datasets.load_boston()
X_boston = boston.data
y_boston = boston.target

# Split the data
X_train_boston, X_test_boston, y_train_boston, y_test_boston = model_selection.train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42
)

# Create and train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_boston, y_train_boston)

# Make predictions
y_pred_boston = lr_model.predict(X_test_boston)

# Evaluate the model
mse = metrics.mean_squared_error(y_test_boston, y_pred_boston)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test_boston, y_pred_boston)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_boston, y_pred_boston, alpha=0.7)
plt.plot([y_boston.min(), y_boston.max()], [y_boston.min(), y_boston.max()], 'k--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Housing Prices')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Question 6: Perform 5-fold cross-validation on a Random Forest classifier using the iris dataset
print("\nSolution to Question 6: Cross-Validation with Random Forest")
# Create a Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)

# Perform 5-fold cross-validation
cv_scores = model_selection.cross_val_score(rf_clf, X, y, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# Question 7: Create a confusion matrix for a classification model of your choice on the iris dataset
print("\nSolution to Question 7: Confusion Matrix")
# Train a Random Forest classifier on the full dataset
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Create and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Random Forest on Iris Dataset')
plt.show()

# Question 8: Implement a simple grid search to find the best parameters for an SVM classifier on the iris dataset
print("\nSolution to Question 8: Grid Search for SVM")
# Define the parameter grid to search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Create an SVM classifier
svc = SVC(random_state=42)

# Create a grid search object
grid_search = model_selection.GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# Fit the grid search
grid_search.fit(X, y)

# Print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Question 9: Calculate and plot the feature importances of a Random Forest classifier trained on the iris dataset
print("\nSolution to Question 9: Feature Importances")
# Train a Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X, y)

# Get feature importances
importances = rf_clf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances from Random Forest')
plt.tight_layout()
plt.show()

print("Feature importances:")
for i in range(X.shape[1]):
    print(f"{iris.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Question 10: Create an ROC curve for a binary classification problem
print("\nSolution to Question 10: ROC Curve")
# Convert to binary classification (versicolor vs others)
y_binary = (y == 1).astype(int)  # 1 for versicolor, 0 for others

# Split the data
X_train_bin, X_test_bin, y_train_bin, y_test_bin = model_selection.train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_bin, y_train_bin)

# Get predicted probabilities
y_proba = log_reg.predict_proba(X_test_bin)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_proba)

# Calculate AUC
auc = roc_auc_score(y_test_bin, y_proba)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Versicolor vs Others')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(f"AUC Score: {auc:.4f}")

print("\nAll solutions have been demonstrated!")# Machine Learning Practice Questions - Solutions
# These are solutions to the practice questions from the Machine Learning tutorial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, classification_report

# Set a random seed for reproducibility
np.random.seed(42)

# Question 1: Load the iris dataset and split it into training and testing sets (80% training, 20% testing)
print("Solution to Question 1: Split Iris Dataset")
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Question 2: Train a Decision Tree classifier on the iris dataset and report its accuracy
print("\nSolution to Question 2: Decision Tree Classifier")
# Create and train a decision tree classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Question 3: Create a scatter plot of the first two features of the iris dataset, coloring points by their species
print("\nSolution to Question 3: Scatter Plot of Iris Features")
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=70, alpha=0.7)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris Dataset: Sepal Length vs Sepal Width')
plt.colorbar(scatter, label='Species')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Question 4: Implement a K-means clustering algorithm on the iris dataset with 3 clusters and visualize the results
print("\nSolution to Question 4: K-means Clustering")
# Create and fit the K-means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster assignments
clusters = kmeans.labels_

# Visualize the clusters (using first two features)
plt.figure(figsize=(12, 5))

# Plot true classes
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('True Classes')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Plot K-means clusters
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
plt.title('K-means Clusters')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()

# Calculate adjusted rand index
ari = metrics.adjusted_rand_score(y, clusters)
print(f"Adjusted Rand Index: {ari:.4f}")

# Question 5: Train a Linear Regression model on the Boston Housing dataset to predict house prices
print("\nSolution to Question 5: Linear Regression on Boston Housing")
# Load the Boston Housing dataset
boston = datasets.load_boston()
X_boston = boston.data
y_boston = boston.target

# Split the data
X_train_boston, X_test_boston, y_train_boston, y_test_boston = model_selection.train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42
)

# Create and train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_boston, y_train_boston)

# Make predictions
y_pred_boston = lr_model.predict(X_test_boston)

# Evaluate the model
mse = metrics.mean_squared_error(y_test_boston, y_pred_boston)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test_boston, y_pred_boston)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_boston, y_pred_boston, alpha=0.7)
plt.plot([y_boston.min(), y_boston.max()], [y_boston.min(), y_boston.max()], 'k--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Housing Prices')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Question 6: Perform 5-fold cross-validation on a Random Forest classifier using the iris dataset
print("\nSolution to Question 6: Cross-Validation with Random Forest")
# Create a Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)

# Perform 5-fold cross-validation
cv_scores = model_selection.cross_val_score(rf_clf, X, y, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# Question 7: Create a confusion matrix for a classification model of your choice on the iris dataset
print("\nSolution to Question 7: Confusion Matrix")
# Train a Random Forest classifier on the full dataset
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Create and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Random Forest on Iris Dataset')
plt.show()

# Question 8: Implement a simple grid search to find the best parameters for an SVM classifier on the iris dataset
print("\nSolution to Question 8: Grid Search for SVM")
# Define the parameter grid to search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Create an SVM classifier
svc = SVC(random_state=42)

# Create a grid search object
grid_search = model_selection.GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# Fit the grid search
grid_search.fit(X, y)

# Print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Question 9: Calculate and plot the feature importances of a Random Forest classifier trained on the iris dataset
print("\nSolution to Question 9: Feature Importances")
# Train a Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X, y)

# Get feature importances
importances = rf_clf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances from Random Forest')
plt.tight_layout()
plt.show()

print("Feature importances:")
for i in range(X.shape[1]):
    print(f"{iris.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Question 10: Create an ROC curve for a binary classification problem
print("\nSolution to Question 10: ROC Curve")
# Convert to binary classification (versicolor vs others)
y_binary = (y == 1).astype(int)  # 1 for versicolor, 0 for others

# Split the data
X_train_bin, X_test_bin, y_train_bin, y_test_bin = model_selection.train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_bin, y_train_bin)

# Get predicted probabilities
y_proba = log_reg.predict_proba(X_test_bin)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_proba)

# Calculate AUC
auc = roc_auc_score(y_test_bin, y_proba)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Versicolor vs Others')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(f"AUC Score: {auc:.4f}")

print("\nAll solutions have been demonstrated!")


