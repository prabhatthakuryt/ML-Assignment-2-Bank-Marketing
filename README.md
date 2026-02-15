# Machine Learning Assignment 2  
## Bank Marketing Classification using Multiple ML Models

---

## 1. Problem Statement

The objective of this project is to predict whether a customer will subscribe to a term deposit based on data collected from a direct marketing campaign conducted by a Portuguese banking institution.

This is a supervised classification problem where the target variable is:

- **y** → Whether the client subscribed to a term deposit (Yes/No)

---

## 2. Dataset Description

- **Dataset Name:** Bank Marketing Dataset
- **Source:** UCI Machine Learning Repository
- **Total Instances:** 45,211
- **Total Features:** 16 input features
- **Target Variable:** y (Binary Classification)

### Feature Types:
- Numerical features: age, balance, duration, campaign, pdays, previous
- Categorical features: job, marital, education, housing, loan, contact, month, poutcome, etc.

The dataset is imbalanced, with the majority of customers not subscribing to the term deposit.

---

## 3. Data Preprocessing Steps

1. Loaded dataset using pandas
2. Applied Label Encoding to all categorical columns
3. Stored encoders for reuse in Streamlit app
4. Split dataset into training and testing sets (80:20)
5. Applied StandardScaler for feature scaling
6. Saved scaler for deployment consistency

---

## 4. Machine Learning Models Implemented

The following 6 classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

## 5. Evaluation Metrics

For each model, the following evaluation metrics were calculated:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 6. Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|----------|------|------------|--------|----------|------|
| Logistic Regression | 0.8879 | 0.8700 | 0.5970 | 0.2172 | 0.3185 | 0.3134 |
| Decision Tree | 0.8708 | 0.6976 | 0.4650 | 0.4693 | 0.4672 | 0.3937 |
| KNN | 0.8912 | 0.8260 | 0.5859 | 0.3346 | 0.4259 | 0.3885 |
| Naive Bayes | 0.8248 | 0.8094 | 0.3427 | 0.4922 | 0.4041 | 0.3121 |
| Random Forest | 0.9009 | 0.9225 | 0.6323 | 0.4271 | 0.5098 | 0.4679 |
| XGBoost | 0.9081 | 0.9285 | 0.6563 | 0.5005 | 0.5679 | 0.5234 |

---

## 7. Model Performance Observations

### Logistic Regression
Achieved good overall accuracy but very low recall, indicating difficulty in identifying the minority class due to class imbalance.

### Decision Tree
Provided more balanced precision and recall compared to Logistic Regression but lower AUC, indicating moderate generalization.

### KNN
Performed moderately well but is sensitive to feature scaling and data distribution.

### Naive Bayes
Higher recall but lower precision, resulting in more false positive predictions.

### Random Forest
Significantly improved AUC and MCC values due to ensemble learning and reduced overfitting.

### XGBoost
Delivered the best overall performance with highest Accuracy, AUC, and MCC. It effectively handled bias-variance tradeoff and class imbalance.

---

## 8. Conclusion

Among all the models implemented, **XGBoost** achieved the best overall performance across most evaluation metrics.

Ensemble methods (Random Forest and XGBoost) outperformed traditional models due to better generalization and robustness.

The dataset’s class imbalance affected recall values for most models, especially linear models.

---

## 9. Streamlit Web Application Features

The deployed Streamlit app includes:

- CSV Dataset Upload Option
- Model Selection Dropdown
- Display of Evaluation Metrics
- Confusion Matrix
- Classification Report

---


