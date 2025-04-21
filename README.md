# 💸 Financial Risk Prediction

This repository contains the implementation and analysis of a machine learning-based approach to predict financial risk. The aim is to empower individuals with insights into their financial health, promoting smarter financial decisions and overall well-being.

## 📌 Overview

Financial literacy is crucial in today’s economy. This project proposes a user-friendly tool to assess financial risk using machine learning. The system analyzes user-provided financial and demographic data to categorize individuals into risk classes (low, medium, high), allowing early intervention and better planning.

---

## 📂 Project Structure

- `models/`: Contains trained models and scripts for Decision Tree, Random Forest, XGBoost, MLP, and SVM.
- `preprocessing/`: Code for handling missing values, encoding categorical data, feature scaling, and SMOTE for class balancing.
- `app/`: A Flask-based web application for real-time predictions.
- `notebooks/`: Jupyter Notebooks used for model training, evaluation, and result visualization.
- `data/`: The Financial Risk Assessment dataset (not included due to privacy).
- `report/`: Contains the final project report in PDF format.

---

## 📊 Dataset

The dataset includes financial and demographic features:
- Income
- Credit Score
- Loan Amount
- Asset Value
- Dependents
- Previous Defaults
- Gender, Marital Status, Education, etc.

**Preprocessing Techniques Used:**
- Missing value imputation using median.
- Label encoding for categorical variables.
- SMOTE for class balancing.
- Feature normalization via `StandardScaler`.

---

## 🧠 Models Used

| Model               | Key Features                                                     |
|--------------------|------------------------------------------------------------------|
| Decision Tree       | Easy to interpret, but prone to overfitting                      |
| Random Forest       | Ensemble of trees, reduced overfitting, better accuracy          |
| XGBoost             | High performance with regularization and boosting                |
| Voting Classifier   | Combines above models for improved stability                     |
| MLP (Neural Net)    | Captures non-linear patterns, needs more data                    |
| SVM                 | Effective in high-dimensional space, sensitive to noise          |

---

## 🔧 Hyperparameters (Selected)

| Model          | Key Hyperparameters                                |
|----------------|-----------------------------------------------------|
| Decision Tree  | `max_depth=9`, `min_samples_split=2`               |
| Random Forest  | `n_estimators=1000`, `max_depth=500`               |
| XGBoost        | `n_estimators=1550`, `max_depth=3`, `lr=0.01`      |
| MLP            | `layers=(150,100,50)`, `activation='tanh'`         |
| SVM            | `kernel='rbf'`, `C=1`, `gamma=0.1`                  |

---

## 🌐 Web Application

A Flask-based interface was developed to allow users to input their financial details and receive instant financial risk predictions. The app integrates the trained model and preprocessing pipeline for real-time evaluation.

---

## 📚 References

1. [Decision Trees](https://doi.org/10.5281/zenodo.1234567)
2. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
3. [Ensemble Methods in ML](https://doi.org/10.1007/3-540-45014-9_1)
4. [Support Vector Machines](https://doi.org/10.1109/5254.708428)
5. [Random Forest in Credit Risk](https://doi.org/10.3166/ISI.27.5.815-829)
6. [Multilayer Perceptron Review](https://wseas.org/multilayer-perceptron)
7. [Ensemble Innovations (LightGBM, XGBoost)](https://arxiv.org/abs/2402.17979)

---

⭐ If you found this useful, consider starring the repo!
