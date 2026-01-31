r# Store Sales Forecasting - Regularized Regression

## Project Overview

**Objective**: Predict store sales using Regularized Regression (Ridge & Lasso) to prevent overfitting and identify key drivers of sales.

**Tech Stack**: Python, Pandas, Scikit-Learn, Matplotlib, Seaborn

**Dataset**: Store Sales - Time Series Forecasting (from Kaggle)

---

## Features

✅ Complete data preprocessing pipeline  
✅ Date feature extraction (year, month, day, day of week)  
✅ Categorical variable encoding (one-hot encoding)  
✅ Feature standardization for regularization  
✅ Three regression models:
   - Linear Regression (baseline)
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)  
✅ Model performance comparison (MAE, RMSE)  
✅ Feature importance analysis  
✅ Comprehensive visualizations

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Download Dataset

1. Visit the Kaggle competition page:  
   **https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data**

2. Download `train.csv`

3. Place `train.csv` in the project folder:
   ```
   Task 1/
   ├── store_sales_forecasting.py
   ├── train.csv  ← Place here
   ├── requirements.txt
   └── README.md
   ```

### 3. Run the Script

```bash
python store_sales_forecasting.py
```

---

## What the Script Does

### Step-by-Step Process:

1. **Data Loading**: Loads and validates the `train.csv` dataset
2. **Preprocessing**:
   - Converts dates to numeric features (year, month, day, day_of_week)
   - One-hot encodes categorical variables (product family)
   - Handles missing values
3. **Data Preparation**:
   - Separates features (X) and target (y)
   - Standardizes features (required for Ridge/Lasso)
   - Splits into training (80%) and test (20%) sets
4. **Model Training**:
   - Trains Linear Regression (baseline)
   - Trains Ridge Regression (alpha=1.0)
   - Trains Lasso Regression (alpha=0.1)
5. **Evaluation**:
   - Compares models using MAE and RMSE
   - Identifies the best-performing model
6. **Feature Importance**:
   - Analyzes Lasso coefficients
   - Shows which features drive sales
   - Counts eliminated features (coefficient = 0)
7. **Visualization**:
   - Model performance comparison
   - Top 10 important features
   - Predicted vs Actual scatter plot
   - Residuals distribution

---

## Expected Output

### Console Output:
```
===============================================================================
STORE SALES FORECASTING - REGULARIZED REGRESSION
===============================================================================

[STEP 1] Loading Dataset...
✓ Dataset loaded successfully!
  Dataset Shape: (3000888, 6)

[STEP 2] Data Preprocessing & Feature Engineering...
  → Converting date column to datetime features...
    ✓ Date features extracted: year, month, day, day_of_week
  → Processing categorical variables...
    ✓ One-hot encoding 'family' column...
    ✓ Categorical encoding complete
  → Checking for missing values...
    ✓ No missing values found

[STEP 3] Preparing Data for Modeling...
  Features shape: (3000888, 41)
  Target shape: (3000888,)

[STEP 4] Standardizing Features...
  ✓ Features standardized (mean=0, variance=1)
  Training set size: 2400710 samples
  Test set size: 600178 samples

[STEP 5] Training Models...
  → Training Linear Regression (Baseline)...
    ✓ Linear Regression trained
  → Training Ridge Regression (L2 Regularization)...
    ✓ Ridge Regression trained (alpha=1.0)
  → Training Lasso Regression (L1 Regularization)...
    ✓ Lasso Regression trained (alpha=0.1)

[STEP 6] Model Evaluation & Comparison
===============================================================================

Mean Absolute Error (MAE) - Lower is Better:
--------------------------------------------------
  Linear Regression MAE: $XXX.XXXX
  Ridge Regression MAE:  $XXX.XXXX
  Lasso Regression MAE:  $XXX.XXXX

Root Mean Squared Error (RMSE):
--------------------------------------------------
  Linear Regression RMSE: $XXX.XXXX
  Ridge Regression RMSE:  $XXX.XXXX
  Lasso Regression RMSE:  $XXX.XXXX

✓ Best Model: Ridge Regression (MAE: $XXX.XXXX)

[STEP 7] Feature Importance Analysis (Lasso)
===============================================================================

  Lasso Feature Selection:
  → Features with non-zero coefficients: XX
  → Features eliminated (coefficient = 0): XX

  Top 10 Most Important Features (by absolute coefficient):
--------------------------------------------------------------------------------
  ↑ feature_name                   Coefficient:     X.XXXX
  ...

[STEP 8] Generating Visualizations...
  ✓ Visualization saved as 'sales_forecasting_analysis.png'

===============================================================================
PROJECT SUMMARY & INSIGHTS
===============================================================================

📊 Key Findings:
--------------------------------------------------------------------------------
  • Ridge regression improved over baseline by X.XX%
  • Lasso regression improved over baseline by X.XX%

  • Lasso identified XX relevant features
  • XX features were eliminated by Lasso (coefficient = 0)

  • Top sales driver: feature_name
    Coefficient: X.XXXX

✓ Analysis complete!
===============================================================================
```

### Generated File:
- **sales_forecasting_analysis.png**: 4-panel visualization showing:
  1. Model performance comparison (bar chart)
  2. Top 10 features (horizontal bar chart)
  3. Predicted vs Actual sales (scatter plot)
  4. Residuals distribution (histogram)

---

## Understanding the Results

### Regularization Benefits:

- **Ridge (L2)**: Reduces coefficient magnitude → prevents overfitting
- **Lasso (L1)**: Can set coefficients to zero → automatic feature selection

### Interpreting Feature Importance:

- **Positive coefficient**: Increases sales
- **Negative coefficient**: Decreases sales
- **Zero coefficient**: Feature is irrelevant (Lasso only)
- **Large absolute value**: Stronger impact on sales

### Model Selection:

If Ridge/Lasso MAE < Linear MAE → Regularization successfully reduced overfitting and should be used in production.

---

## Customization & Tuning

### Adjust Regularization Strength:

In [store_sales_forecasting.py](store_sales_forecasting.py#L117-L127), modify the `alpha` parameters:

```python
# Stronger regularization (more penalty):
ridge = Ridge(alpha=10.0)
lasso = Lasso(alpha=1.0)

# Weaker regularization (less penalty):
ridge = Ridge(alpha=0.1)
lasso = Lasso(alpha=0.01)
```

### Encode Store Number as Categorical:

In [store_sales_forecasting.py](store_sales_forecasting.py#L68), uncomment:

```python
df = pd.get_dummies(df, columns=['store_nbr'], drop_first=True)
```

### Use Time Series Split:

For more realistic time series validation, replace the train-test split:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Train and evaluate models...
```

---

## Troubleshooting

### Issue: FileNotFoundError
**Solution**: Make sure `train.csv` is in the same directory as the script.

### Issue: ConvergenceWarning (Lasso)
**Solution**: Increase `max_iter` parameter:
```python
lasso = Lasso(alpha=0.1, max_iter=10000)
```

### Issue: Memory Error
**Solution**: Use a subset of data:
```python
df = df.sample(n=100000, random_state=42)  # Use 100k samples
```

---

## Project Structure

```
Task 1/
├── store_sales_forecasting.py    # Main script
├── train.csv                      # Dataset (download separately)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── sales_forecasting_analysis.png # Generated visualization (after running)
```

---

## References

- **Dataset**: [Kaggle - Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Scikit-Learn**: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) | [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- **Documentation**: Python 3.9+, Pandas, NumPy, Matplotlib, Seaborn

---

## License

This project is for educational purposes as part of Advanced Task 1.

---

## Author

Advanced Task 1 - Store Sales Forecasting
Date: January 31, 2026
