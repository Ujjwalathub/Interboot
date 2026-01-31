# 📊 Store Sales Forecasting (Advanced Internship Project)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

## 📌 Project Overview
This repository contains the complete submission for the **Internboot Data Analytics Internship (Advanced Level)**. The project focuses on predicting daily store sales using Machine Learning techniques, specifically Regularized Regression, and deploying the solution as an interactive web application.

## 📂 Tasks Breakdown

### ✅ Task 1: Regularized Regression (Ridge & Lasso)
* **Objective:** Prevent overfitting and identify key sales drivers.
* **Method:** Compared Linear, Ridge, and Lasso Regression.
* **Result:** Lasso Regression was the most effective, identifying "Product Family" and "Promotions" as the top factors driving sales.
* **File:** `task1_lasso.py`

### ✅ Task 2: Regression with External Data
* **Objective:** Improve accuracy by incorporating economic data.
* **Method:** Merged daily oil price data (`oil.csv`) into the training set to account for economic fluctuations.
* **Result:** Analyzed the correlation between oil prices and retail sales to refine the model's predictive power.
* **File:** `task2_external_data.py`

### ✅ Task 3: Model Deployment (Dashboard)
* **Objective:** Create a user-friendly interface for the model.
* **Method:** Built a **Streamlit** web application.
* **Features:**
    * Interactive sliders for Oil Price and Promotions.
    * Real-time sales prediction.
    * Visual graphs of predicted scenarios.
* **File:** `app.py`

---

## 🚀 How to Run the Project

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install -r requirements.txt
