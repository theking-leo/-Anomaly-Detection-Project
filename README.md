# -Anomaly-Detection-Project
Anomaly Detection in Credit Card Transactions
This project focuses on detecting anomalies or outliers in credit card transaction data, which is crucial for identifying potential fraud. The following guide breaks down each step in the process, providing detailed explanations and code snippets.

Table of Contents
Introduction
Prerequisites
Project Structure
Steps
Step 1: Import Libraries and Load Data
Step 2: Explore the Data
Step 3: Data Visualization
Step 4: Data Preprocessing
Step 5: Split the Data
Step 6: Train an Anomaly Detection Model
Step 7: Evaluate the Model
Step 8: Interpret the Results
Advanced Techniques
Conclusion
Introduction
Anomaly detection is essential in various domains, such as fraud detection, network security, and sensor data monitoring. This project demonstrates how to detect anomalies in a dataset of credit card transactions using different machine learning techniques.

Prerequisites
Python 3.x
Libraries: pandas, scikit-learn, matplotlib, seaborn, imbalanced-learn, keras, numpy
Install the necessary libraries using:

bash
Copy code
pip install pandas scikit-learn matplotlib seaborn imbalanced-learn keras numpy
Project Structure
creditcard.csv: The dataset containing credit card transactions.
anomaly_detection.py: The main script for running the anomaly detection process.
classification_report.csv: The output file containing the classification report.
Steps
Step 1: Import Libraries and Load Data
Import the necessary libraries and load the dataset.

python
Copy code
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('/Users/aghgang/Downloads/personalproject/creditcard.csv')

# Display the first few rows of the dataset
print(data.head())
Step 2: Explore the Data
Explore the dataset to understand its structure and check for missing values.

python
Copy code
# Check the shape of the data
print(f"Data Shape: {data.shape}")

# Get basic information about the data
print(data.info())

# Check for missing values
print(f"Missing values in each column:\n{data.isnull().sum()}")
Step 3: Data Visualization
Visualize the data to understand its distribution and identify any patterns.

python
Copy code
# Plot the distribution of the 'Amount' column
plt.figure(figsize=(10, 6))
sns.histplot(data['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Plot the count of fraudulent vs non-fraudulent transactions
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=data)
plt.title('Count of Fraudulent vs Non-Fraudulent Transactions')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()
Step 4: Data Preprocessing
Preprocess the data by scaling the features.

python
Copy code
# Scaling the 'Amount' and 'Time' features
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Display the first few rows of the preprocessed data
print(data.head())
Step 5: Split the Data
Split the data into training and testing sets.

python
Copy code
# Define features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
Step 6: Train an Anomaly Detection Model
Train the Isolation Forest model to detect anomalies.

python
Copy code
# Train the Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train)

# Predict anomalies in the test set
y_pred = model.predict(X_test)
Step 7: Evaluate the Model
Evaluate the model's performance using metrics like accuracy, precision, recall, and the F1 score.

python
Copy code
# Convert predictions to binary output
y_pred_binary = [1 if x == -1 else 0 for x in y_pred]

# Print the classification report
print(classification_report(y_test, y_pred_binary))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
Step 8: Interpret the Results
Interpret the results to understand the effectiveness of the model.

python
Copy code
# Generate and display the classification report
report = classification_report(y_test, y_pred_binary, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Print the classification report
print(report_df)

# Save the report to a CSV file
report_df.to_csv('/Users/aghgang/Downloads/personalproject/classification_report.csv')
Advanced Techniques
Handling Imbalanced Data Using SMOTE
Use SMOTE to handle imbalanced data.

python
Copy code
from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scale the data
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Train the Isolation Forest model on resampled data
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train_res_scaled)

# Predict anomalies in the test set
y_pred = model.predict(X_test_scaled)

# Convert predictions to binary output
y_pred_binary = [1 if x == -1 else 0 for x in y_pred]

# Display the classification report
print(classification_report(y_test, y_pred_binary))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
Using One-Class SVM for Anomaly Detection
Try using One-Class SVM for better results.

python
Copy code
from sklearn.svm import OneClassSVM

# Train the One-Class SVM model
ocsvm_model = OneClassSVM(kernel='rbf', nu=0.01, gamma='auto')
ocsvm_model.fit(X_train_res_scaled)

# Predict anomalies in the test set
y_pred = ocsvm_model.predict(X_test_scaled)

# Convert predictions to binary output
y_pred_binary = [1 if x == -1 else 0 for x in y_pred]

# Display the classification report
print(classification_report(y_test, y_pred_binary))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
Conclusion
Improving anomaly detection models for imbalanced datasets like fraud detection requires multiple approaches and fine-tuning. Techniques such as SMOTE, feature engineering, and using different algorithms can significantly enhance performance. Continue experimenting with these methods to find the best solution for your data.
