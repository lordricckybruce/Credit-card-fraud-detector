#!/bin/python3
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score

# Step 2: Load the dataset
# This is a typical credit card fraud detection dataset, replace this path with your dataset
data = pd.read_csv('credit_card_transactions.csv')

# Step 3: Explore the dataset
# You can check the first few rows of the dataset to see what it looks like
print(data.head())

# Step 4: Preprocessing the data
# Split features and target (label: 1 for fraud, 0 for non-fraud)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable (fraud or not)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for many models, like Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train a Random Forest Classifier (you can choose other models like Decision Tree, etc.)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model on the test set
y_pred = model.predict(X_test)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Confusion Matrix
print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")

# Classification Report (Precision, Recall, F1-score)
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

# ROC-AUC Score (best for imbalanced datasets)
print(f"ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")

