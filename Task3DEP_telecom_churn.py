import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'telecom_churn.csv'
data = pd.read_csv(file_path)

print(data.info())

print(data.describe())

plt.figure(figsize=(10, 6))

sns.countplot(x='Churn', data=data)
plt.title('Distribution of Churn')
plt.show()

data.hist(bins=30, figsize=(20, 15))
plt.show()

plt.figure(figsize=(15, 10))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.title('Boxplots of Numerical Features')
plt.show()

plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

X = data.drop('Churn', axis=1)
y = data['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Evaluation for Logistic Regression
log_reg_report = classification_report(y_test, y_pred_log_reg)
print("Logistic Regression Report:\n", log_reg_report)
log_reg_cm = confusion_matrix(y_test, y_pred_log_reg)
print("Logistic Regression Confusion Matrix:\n", log_reg_cm)
log_reg_acc = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy: ", log_reg_acc)

# Evaluation for Random Forest Classifier
rf_report = classification_report(y_test, y_pred_rf)
print("Random Forest Report:\n", rf_report)
rf_cm = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:\n", rf_cm)
rf_acc = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy: ", rf_acc)
