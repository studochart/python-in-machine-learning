"""
Supervised Learning Practice Questions - Solutions

This file contains solutions to the practice questions from the supervised learning blog post.
Each solution is well-commented to explain the approach and key concepts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("SUPERVISED LEARNING PRACTICE SOLUTIONS\n")

# -------------------------------------------------------------------------
# Question 1: Load the iris dataset and split it into training and testing sets
# -------------------------------------------------------------------------

print("\n--- Question 1: Loading and Splitting the Iris Dataset ---")

# Load the iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species of iris (0=setosa, 1=versicolor, 2=virginica)
feature_names = iris.feature_names
target_names = iris.target_names

# Print dataset information
print(f"Dataset shape: {X.shape}")  # 150 samples, 4 features
print(f"Number of classes: {len(np.unique(y))}")  # 3 classes
print(f"Class distribution: {np.bincount(y)}")  # 50 samples per class

# Split the data into training (70%) and testing (30%) sets
# random_state ensures reproducibility of results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Print the shapes of the resulting sets
print(f"Training set shape: {X_train.shape}")  # 105 samples (70% of 150)
print(f"Testing set shape: {X_test.shape}")  # 45 samples (30% of 150)

# -------------------------------------------------------------------------
# Question 2: Train a Decision Tree classifier on the iris dataset
# -------------------------------------------------------------------------

print("\n--- Question 2: Training a Decision Tree Classifier ---")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize the Decision Tree classifier
# random_state ensures reproducibility of results
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on our training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy (proportion of correct predictions)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree accuracy: {accuracy:.4f}")

# Generate a detailed classification report
# This shows precision, recall, and F1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# -------------------------------------------------------------------------
# Question 3: Compare different classification algorithms
# -------------------------------------------------------------------------

print("\n--- Question 3: Comparing Classification Algorithms ---")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Initialize different classifiers
# We'll compare 5 popular classification algorithms
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200)
}

# Train and evaluate each classifier
results = {}
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name} accuracy: {accuracy:.4f}")

# Visualize the results with a bar chart
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Algorithms on Iris Dataset')
plt.ylim(0.8, 1.0)  # Set y-axis limits for better visualization
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Find the best classifier
best_classifier = max(results, key=results.get)
print(f"\nBest classifier: {best_classifier} with accuracy {results[best_classifier]:.4f}")

# -------------------------------------------------------------------------
# Question 4: Implement grid search for hyperparameter tuning
# -------------------------------------------------------------------------

print("\n--- Question 4: Grid Search for Random Forest ---")

from sklearn.model_selection import GridSearchCV

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid to search
# We'll try different values for:
# - n_estimators: number of trees in the forest
# - max_depth: maximum depth of each tree
# - min_samples_split: minimum samples required to split a node
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create the grid search with 5-fold cross-validation
# This will try all combinations of parameters (3×3×3 = 27 combinations)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)

# Perform the grid search (this may take a while)
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate on the test set using the best parameters
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy with best parameters: {test_accuracy:.4f}")

# Compare with default parameters
default_rf = RandomForestClassifier(random_state=42)
default_rf.fit(X_train, y_train)
default_y_pred = default_rf.predict(X_test)
default_accuracy = accuracy_score(y_test, default_y_pred)
print(f"Test accuracy with default parameters: {default_accuracy:.4f}")

# -------------------------------------------------------------------------
# Question 5: Create and interpret a confusion matrix
# -------------------------------------------------------------------------

print("\n--- Question 5: Confusion Matrix Analysis ---")

from sklearn.metrics import confusion_matrix

# Train a Random Forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Create the confusion matrix
# Rows represent actual classes, columns represent predicted classes
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Interpret the results
print("\nConfusion Matrix Interpretation:")

# Calculate metrics for each class
for i, class_name in enumerate(target_names):
    # Extract values from confusion matrix
    true_positives = conf_matrix[i, i]
    false_positives = conf_matrix[:, i].sum() - true_positives
    false_negatives = conf_matrix[i, :].sum() - true_positives
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print(f"Class: {class_name}")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

# -------------------------------------------------------------------------
# Question 6: Train a Linear Regression model on the Boston Housing dataset
# -------------------------------------------------------------------------

print("\n--- Question 6: Linear Regression on Boston Housing Dataset ---")

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
# This dataset contains information about housing in Boston
# The target is the median value of owner-occupied homes in $1000s
boston = load_boston()
X_boston = boston.data
y_boston = boston.target
feature_names_boston = boston.feature_names

# Print dataset information
print(f"Dataset shape: {X_boston.shape}")  # 506 samples, 13 features
print(f"Feature names: {feature_names_boston}")
print(f"Target range: {y_boston.min():.2f} to {y_boston.max():.2f}")

# Split the data into training (80%) and testing (20%) sets
X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42
)

# Train a Linear Regression model
# Linear Regression models the relationship as y = mx + b
model = LinearRegression()
model.fit(X_train_boston, y_train_boston)

# Make predictions on the test set
y_pred_boston = model.predict(X_test_boston)

# Evaluate the model using regression metrics
# MSE: Mean Squared Error - average of squared differences
# RMSE: Root Mean Squared Error - square root of MSE, in same units as target
# R²: Coefficient of Determination - proportion of variance explained by model
mse = mean_squared_error(y_test_boston, y_pred_boston)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_boston, y_pred_boston)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_boston, y_pred_boston, alpha=0.5)
plt.plot([y_test_boston.min(), y_test_boston.max()], 
         [y_test_boston.min(), y_test_boston.max()], 'r--')
plt.xlabel('Actual Prices ($1000s)')
plt.ylabel('Predicted Prices ($1000s)')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.show()

# Analyze feature coefficients to understand their impact
coef = pd.DataFrame({
    'Feature': feature_names_boston,
    'Coefficient': model.coef_
})
coef = coef.sort_values('Coefficient', ascending=False)

print("\nFeature Coefficients:")
print(coef)

# Visualize feature coefficients
plt.figure(figsize=(12, 6))
plt.barh(coef['Feature'], coef['Coefficient'])
plt.xlabel('Coefficient')
plt.title('Feature Coefficients')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Question 7: Implement feature scaling and evaluate its impact
# -------------------------------------------------------------------------

print("\n--- Question 7: Impact of Feature Scaling ---")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

# We'll use the Boston Housing dataset again
# Split the data into training and testing sets
X_train_scale, X_test_scale, y_train_scale, y_test_scale = train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42
)

# Train models without scaling
models_unscaled = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=0.1, random_state=42)
}

results_unscaled = {}
for name, model in models_unscaled.items():
    # Train the model on unscaled data
    model.fit(X_train_scale, y_train_scale)
    y_pred = model.predict(X_test_scale)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_scale, y_pred)
    r2 = r2_score(y_test_scale, y_pred)
    results_unscaled[name] = {'MSE': mse, 'R²': r2}

# Apply feature scaling using StandardScaler
# This transforms features to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_scale)
X_test_scaled = scaler.transform(X_test_scale)

# Train models with scaling
models_scaled = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=0.1, random_state=42)
}

results_scaled = {}
for name, model in models_scaled.items():
    # Train the model on scaled data
    model.fit(X_train_scaled, y_train_scale)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_scale, y_pred)
    r2 = r2_score(y_test_scale, y_pred)
    results_scaled[name] = {'MSE': mse, 'R²': r2}

# Compare results
print("Results without scaling:")
for name, metrics in results_unscaled.items():
    print(f"  {name}: MSE = {metrics['MSE']:.4f}, R² = {metrics['R²']:.4f}")

print("\nResults with scaling:")
for name, metrics in results_scaled.items():
    print(f"  {name}: MSE = {metrics['MSE']:.4f}, R² = {metrics['R²']:.4f}")

# Visualize the comparison of MSE values
models = list(results_unscaled.keys())
mse_unscaled = [results_unscaled[model]['MSE'] for model in models]
mse_scaled = [results_scaled[model]['MSE'] for model in models]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, mse_unscaled, width, label='Without Scaling')
plt.bar(x + width/2, mse_scaled, width, label='With Scaling')
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('Impact of Feature Scaling on Model Performance')
plt.xticks(x, models)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Question 8: Handle an imbalanced dataset using resampling techniques
# -------------------------------------------------------------------------

print("\n--- Question 8: Handling Imbalanced Data ---")

from imblearn.over_sampling import SMOTE
from collections import Counter

# Create an imbalanced dataset by modifying the iris dataset
# We'll keep all samples of class 0, 80% of class 1, and 20% of class 2
indices_0 = np.where(y == 0)[0]
indices_1 = np.where(y == 1)[0]
indices_2 = np.where(y == 2)[0]

np.random.shuffle(indices_1)
np.random.shuffle(indices_2)

indices_1_subset = indices_1[:int(0.8 * len(indices_1))]
indices_2_subset = indices_2[:int(0.2 * len(indices_2))]

imbalanced_indices = np.concatenate([indices_0, indices_1_subset, indices_2_subset])
X_imbalanced = X[imbalanced_indices]
y_imbalanced = y[imbalanced_indices]

# Print class distribution
print("Original class distribution:")
print(np.bincount(y))  # 50 samples per class

print("\nImbalanced class distribution:")
print(np.bincount(y_imbalanced))  # Imbalanced distribution

# Split the imbalanced data
# stratify=y_imbalanced ensures that the class distribution is preserved in both sets
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imbalanced, y_imbalanced, test_size=0.3, random_state=42, stratify=y_imbalanced
)

# Train a classifier on the imbalanced data
clf_imbalanced = RandomForestClassifier(random_state=42)
clf_imbalanced.fit(X_train_imb, y_train_imb)
y_pred_imbalanced = clf_imbalanced.predict(X_test_imb)

# Evaluate on imbalanced data
print("\nResults on imbalanced data:")
print(classification_report(y_test_imb, y_pred_imbalanced, target_names=target_names))

# Apply SMOTE to balance the training data
# SMOTE creates synthetic samples of the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imb, y_train_imb)

# Print resampled class distribution
print("\nResampled class distribution:")
print(Counter(y_train_resampled))  # Equal distribution after SMOTE

# Train a classifier on the resampled data
clf_resampled = RandomForestClassifier(random_state=42)
clf_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = clf_resampled.predict(X_test_imb)

# Evaluate on resampled data
print("\nResults after SMOTE resampling:")
print(classification_report(y_test_imb, y_pred_resampled, target_names=target_names))

# Visualize the confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

conf_matrix_imbalanced = confusion_matrix(y_test_imb, y_pred_imbalanced)
sns.heatmap(conf_matrix_imbalanced, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=ax1)
ax1.set_xlabel('Predicted Labels')
ax1.set_ylabel('True Labels')
ax1.set_title('Confusion Matrix - Imbalanced Data')

conf_matrix_resampled = confusion_matrix(y_test_imb, y_pred_resampled)
sns.heatmap(conf_matrix_resampled, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=ax2)
ax2.set_xlabel('Predicted Labels')
ax2.set_ylabel('True Labels')
ax2.set_title('Confusion Matrix - After SMOTE')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Question 9: Perform feature selection
# -------------------------------------------------------------------------

print("\n--- Question 9: Feature Selection ---")

from sklearn.feature_selection import SelectKBest, f_classif

# We'll use the original iris dataset
# Split the data
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Method 1: Feature selection using SelectKBest (ANOVA F-value)
# This selects features based on their relationship with the target variable
selector = SelectKBest(f_classif, k=2)  # Select the 2 best features
X_train_selected = selector.fit_transform(X_train_fs, y_train_fs)
X_test_selected = selector.transform(X_test_fs)

# Get the scores and p-values for each feature
scores = selector.scores_
p_values = selector.pvalues_

# Print feature scores
print("Feature scores (ANOVA F-value):")
for i, (score, p_value) in enumerate(zip(scores, p_values)):
    print(f"  {feature_names[i]}: Score = {score:.4f}, p-value = {p_value:.4f}")

# Get selected feature indices
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]
print(f"\nSelected features: {selected_features}")

# Method 2: Feature importance from Random Forest
# This uses the model's own measure of feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_fs, y_train_fs)

# Get feature importances
importances = rf.feature_importances_

# Print feature importances
print("\nFeature importances from Random Forest:")
for i, importance in enumerate(importances):
    print(f"  {feature_names[i]}: {importance:.4f}")

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances from Random Forest')
plt.tight_layout()
plt.show()

# Compare model performance with all features vs. selected features
# Train and evaluate with all features
rf_all = RandomForestClassifier(random_state=42)
rf_all.fit(X_train_fs, y_train_fs)
y_pred_all = rf_all.predict(X_test_fs)
accuracy_all = accuracy_score(y_test_fs, y_pred_all)

# Train and evaluate with selected features
rf_selected = RandomForestClassifier(random_state=42)
rf_selected.fit(X_train_selected, y_train_fs)
y_pred_selected = rf_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test_fs, y_pred_selected)

print(f"\nAccuracy with all features: {accuracy_all:.4f}")
print(f"Accuracy with selected features: {accuracy_selected:.4f}")

# -------------------------------------------------------------------------
# Question 10: Implement cross-validation
# -------------------------------------------------------------------------

print("\n--- Question 10: Cross-Validation ---")

from sklearn.model_selection import cross_val_score

# We'll use the original iris dataset
# Define the classifiers to evaluate
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200)
}

# Perform k-fold cross-validation
# In k-fold cross-validation, the data is divided into k equal parts (folds)
# The model is trained k times, each time using k-1 folds for training and 1 fold for validation
k = 5
print(f"Performing {k}-fold cross-validation:")

cv_results = {}
for name, clf in classifiers.items():
    # Perform cross-validation
    scores = cross_val_score(clf, X, y, cv=k, scoring='accuracy')
    
    # Store results
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    
    print(f"  {name}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}")

# Visualize cross-validation results
plt.figure(figsize=(12, 6))

# Plot mean accuracy with error bars
means = [cv_results[name]['mean'] for name in classifiers.keys()]
stds = [cv_results[name]['std'] for name in classifiers.keys()]

plt.bar(classifiers.keys(), means, yerr=stds, capsize=10, alpha=0.7)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title(f'{k}-Fold Cross-Validation Results')
plt.ylim(0.8, 1.0)  # Set y-axis limits for better visualization
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare cross-validation with single train-test split
print("\nComparison with single train-test split:")

# Split the data
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
    X, y, test_size=0.3, random_state=42
)

for name, clf in classifiers.items():
    # Train and evaluate on single split
    clf.fit(X_train_cv, y_train_cv)
    y_pred = clf.predict(X_test_cv)
    accuracy = accuracy_score(y_test_cv, y_pred)
    
    # Compare with cross-validation
    cv_mean = cv_results[name]['mean']
    
    print(f"  {name}: Single split = {accuracy:.4f}, Cross-validation = {cv_mean:.4f}")

print("\nAll practice questions completed!")
