# 🎓 SmartFee : An Intelligent Student Fee Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

---

## 📌 Project Overview

Educational institutions often follow a fixed fee structure or provide fee concessions using predefined rules. However, these traditional approaches may not accurately reflect the financial condition of every student's family, potentially leaving deserving students without assistance.

This project aims to develop a Machine Learning-based classification system that predicts whether a student should **pay the full fee** or **receive a fee concession** based on financial, family, and academic information.

The proposed system acts as a **decision support tool** to help educational institutions make transparent, consistent, and data-driven fee assistance decisions.

---

## 🎯 Problem Statement & Objective

### Problem Statement
Many deserving students are unable to receive financial assistance because traditional fee concession systems often rely on limited criteria. 

This project addresses this challenge by building an AI model that considers multiple socio-economic and academic factors before recommending whether a student should pay the full tuition fee. The objective is to improve fairness, consistency, and transparency in fee concession decisions.

### Primary Objective
The primary objective of this project is to develop a Binary Classification Model that predicts whether a student should:
- ✅ **Pay the Full Fee** (`Yes`)
- ❌ **Receive a Fee Concession** (`No`)

using financial, demographic, and academic information. The project also aims to compare multiple machine learning algorithms and identify the best-performing model.

---

## 📂 Dataset Overview

- **Total Records:** 10,000
- **Total Features:** 16
- **Problem Type:** Binary Classification
- **Target Variable:** `Pay_Full_Fee` (`Yes` / `No`)

The dataset contains both numerical and categorical features that represent the financial condition of a student's family and selected academic information.

### 📊 Dataset Features

| Feature | Description |
|---|---|
| `Student_ID` | Unique identifier |
| `Monthly_Income` | Family's monthly income |
| `Household_Size` | Total family members |
| `Number_of_Earning_Members` | Number of earning members |
| `Number_of_School_Going_Children` | Children currently studying |
| `Savings` | Estimated monthly savings |
| `Medical_Expenses` | Monthly medical expenditure |
| `Attendance` | Student attendance percentage |
| `Academic_Performance` | Student academic score |
| `Teacher_Evaluation` | Teacher rating (1–5) |
| `Income_Stability` | Stable or Fluctuating income |
| `House_Type` | Owned or Rented |
| `Location` | Rural, Semi-Urban, Urban |
| `Parent_Education` | Parent's education level |
| `Debt_Status` | Existing debt status |
| `Pay_Full_Fee` **(Target)** | Target variable (`Yes` / `No`) |

---

## 📌 Feature Importance & Influence Matrix

The dataset has been designed so that different features influence the final decision differently:

### 📈 Positive Influence
*Increases the probability that a student will pay the full fee (`Yes`):*
- Higher Monthly Income
- Higher Savings
- More Earning Members
- Stable Income
- Owned House
- Higher Parent Education

### 📉 Negative Influence
*Increases the probability of receiving a fee concession (`No`):*
- Larger Household Size
- High Medical Expenses
- Active Debt Status
- More School-Going Children
- Lower Income

---

## ⚖️ Metric Strategy: Equity vs. Financial Sustainability

When evaluating model predictions, error types carry vastly different real-world consequences:

* 🚨 **False Positive (FP):** Predicting a student *pays full fee* when they actually *needed a concession*.
  * **Impact (High Risk):** A financially struggling student is forced to pay full tuition and may drop out due to financial burden.
* ⚠️ **False Negative (FN):** Predicting a student *gets a concession* when they actually *should pay full fee*.
  * **Impact (Low Risk):** The institution loses a bit of tuition revenue, but deserving/in-need students get help.

### Why Recall & F1-Score Matter
1. **Prioritizing Recall (Focus on Equity & Assistance):** If the goal is to ensure no needy student is mistakenly denied a concession, minimizing False Positives is critical. Maximizing Recall for the concession group ensures deserving students are caught by the safety net.
2. **Balancing with F1-Score (Focus on Financial Sustainability):** If you only maximize Recall, the model might grant concessions to almost everyone, hurting the institution's operational budget. **F1-Score** gives the harmonic mean of Precision and Recall, keeping a clean balance between protecting needy students and maintaining fiscal sustainability.

---

## 📋 Project Workflow

This project follows a complete Machine Learning pipeline for binary classification:

1. **📌 1. Data Loading:** Import the dataset into Python using **Pandas**, display initial records, and audit structure.
2. **📌 2. Exploratory Data Analysis (EDA):** Analyze distributions of numerical and categorical features, visualize class distributions, and identify correlation patterns.
3. **📌 3. Data Preprocessing:** Handle missing values (`SimpleImputer`), check for duplicates, encode categorical variables (`LabelEncoder`/`OneHotEncoder`), and scale numerical features (`StandardScaler`).
4. **📌 4. Feature Engineering:** Analyze feature importances and correlations to select relevant predictors for training (`X` and `y`).
5. **📌 5. Data Splitting:** Partition data into **training** and **testing** sets using `train_test_split`.
6. **📌 6. Model Training:** Train and compare multiple algorithms:
   - Logistic Regression
   - $k$-Nearest Neighbors (KNN)
   - Naïve Bayes (`GaussianNB`)
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - XGBoost Classifier
   - CatBoost Classifier
   - Adaboost classifier
     
7. **📌 7. Hyperparameter Tuning:** Optimize model performance using `GridSearchCV` and `RandomizedSearchCV`.
8. **📌 8. Model Evaluation:** Evaluate models using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.
9. **📌 9. Conclusion & Recommendations:** Summarize findings, select the best model balancing recall and precision, and suggest potential deployment strategies.

---

## 🛠️ Code Snippet / Pipeline Setup

```python
# 1. Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 2. Loading the Data
df = pd.read_csv("student_data.csv")

# Quick Inspection
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe())

📜 License
This project is open-source and available under the MIT License.
