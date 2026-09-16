# 💳 Credit Card Approval Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether a credit card application belongs to the approved or rejected class using Machine Learning.

The project uses the UCI Credit Approval Dataset and implements data preprocessing, feature transformation, multiple classification algorithms, model evaluation, and a Streamlit-based prediction application.

## 🎯 Objectives

- Analyze the credit approval dataset
- Handle missing values
- Process numerical and categorical features
- Train multiple Machine Learning classification models
- Compare model performance
- Evaluate the final model using Accuracy and ROC-AUC
- Build a user-friendly Streamlit application

## 📊 Dataset

**Dataset:** UCI Credit Approval Dataset

- Instances: 690
- Input Attributes: 15
- Target Classes: `+` and `-`
- Data Types: Numerical and Categorical
- Missing Values: Present

**Dataset Source:**

https://archive.ics.uci.edu/dataset/27/credit%2Bapproval

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- JupyterLab

## 🔧 Data Preprocessing

The following preprocessing techniques were used:

1. Missing values were identified and handled.
2. Numerical missing values were replaced using median imputation.
3. Categorical missing values were replaced using most-frequent imputation.
4. Categorical features were converted using One-Hot Encoding.
5. Numerical features were standardized using StandardScaler.
6. The dataset was divided into training and testing sets using an 80:20 split.

## 🤖 Machine Learning Models

The following classification models were implemented:

- Logistic Regression
- Decision Tree
- Random Forest

## 📈 Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 89.13% |
| Decision Tree | 89.13% |
| Random Forest | 87.68% |

The Logistic Regression model achieved an ROC-AUC score of approximately **0.96** on the test set.

## 📁 Project Structure

```text
Credit-Card-Approval-Prediction/
│
├── Credit_Card_Fraud_Detection.ipynb
├── crx.data
├── credit_card_approval_model.pkl
├── app.py
├── requirements.txt
└── README.md
