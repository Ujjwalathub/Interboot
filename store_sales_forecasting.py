"""
Store Sales Forecasting using Regularized Regression (Ridge & Lasso)
Project: Advanced Task 1
Dataset: Store Sales - Time Series Forecasting (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("STORE SALES FORECASTING - REGULARIZED REGRESSION")
print("=" * 80)

# ============================================================================
# 1. DATA ACQUISITION & LOADING
# ============================================================================
print("\n[STEP 1] Loading Dataset...")

try:
    df = pd.read_csv(r"E:\Project\Data\train.csv")
    print(f"✓ Dataset loaded successfully!")
    print(f"  Dataset Shape: {df.shape}")
    print(f"\n  First few rows:")
    print(df.head())
    print(f"\n  Dataset Info:")
    print(df.info())
except FileNotFoundError:
    print("❌ ERROR: 'train.csv' not found!")
    print("   Please download the dataset from Kaggle:")
    print("   https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data")
    print("   Place 'train.csv' in the same directory as this script.")
    exit(1)

# ============================================================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 2] Data Preprocessing & Feature Engineering...")

# Step 2.1: Handle Dates
print("  → Converting date column to datetime features...")
df['date'] = pd.to_datetime(df['date'])

# Extract numerical features from the date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6

# Drop the original 'date' column
df = df.drop('date', axis=1)
print("    ✓ Date features extracted: year, month, day, day_of_week")

# Step 2.2: Handle Categorical Variables
print("  → Processing categorical variables...")

# Drop 'id' as it provides no predictive value
if 'id' in df.columns:
    df = df.drop('id', axis=1)
    print("    ✓ Dropped 'id' column")

# One-Hot Encode categorical columns ('family')
print("    ✓ One-hot encoding 'family' column...")
df = pd.get_dummies(df, columns=['family'], drop_first=True)

# Note: Treating 'store_nbr' as numeric for simplicity
# In production, you might want to encode it as categorical too
print("    ✓ Categorical encoding complete")

# Step 2.3: Handle Missing Values
print("  → Checking for missing values...")
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    print(f"    ⚠ Found {missing_counts.sum()} missing values")
    print(missing_counts[missing_counts > 0])
    df = df.dropna()
    print("    ✓ Missing values handled")
else:
    print("    ✓ No missing values found")

print(f"\n  Final processed dataset shape: {df.shape}")

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================
print("\n[STEP 3] Preparing Data for Modeling...")

# Define Features (X) and Target (y)
X = df.drop('sales', axis=1)  # All columns except 'sales'
y = df['sales']               # The target we want to predict

print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# ============================================================================
# 4. STANDARDIZATION
# ============================================================================
print("\n[STEP 4] Standardizing Features...")
print("  (Critical for Ridge/Lasso: penalties are scale-sensitive)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("  ✓ Features standardized (mean=0, variance=1)")

# Split into Train (80%) and Test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"  Training set size: {X_train.shape[0]} samples")
print(f"  Test set size: {X_test.shape[0]} samples")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n[STEP 5] Training Models...")

# A. Linear Regression (Baseline)
print("  → Training Linear Regression (Baseline)...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("    ✓ Linear Regression trained")

# B. Ridge Regression (L2 Regularization)
print("  → Training Ridge Regression (L2 Regularization)...")
ridge = Ridge(alpha=1.0)  # You can tune 'alpha' (strength of penalty) later
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("    ✓ Ridge Regression trained (alpha=1.0)")

# C. Lasso Regression (L1 Regularization)
print("  → Training Lasso Regression (L1 Regularization)...")
lasso = Lasso(alpha=0.1)  # Alpha needs tuning
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("    ✓ Lasso Regression trained (alpha=0.1)")

# ============================================================================
# 6. EVALUATION & COMPARISON
# ============================================================================
print("\n[STEP 6] Model Evaluation & Comparison")
print("=" * 80)

# Calculate MAE (Mean Absolute Error)
lr_mae = mean_absolute_error(y_test, lr_pred)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
lasso_mae = mean_absolute_error(y_test, lasso_pred)

print("\nMean Absolute Error (MAE) - Lower is Better:")
print("-" * 50)
print(f"  Linear Regression MAE: ${lr_mae:,.4f}")
print(f"  Ridge Regression MAE:  ${ridge_mae:,.4f}")
print(f"  Lasso Regression MAE:  ${lasso_mae:,.4f}")

# Calculate RMSE (Root Mean Squared Error)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

print("\nRoot Mean Squared Error (RMSE):")
print("-" * 50)
print(f"  Linear Regression RMSE: ${lr_rmse:,.4f}")
print(f"  Ridge Regression RMSE:  ${ridge_rmse:,.4f}")
print(f"  Lasso Regression RMSE:  ${lasso_rmse:,.4f}")

# Determine best model
best_model = min(
    [("Linear", lr_mae), ("Ridge", ridge_mae), ("Lasso", lasso_mae)],
    key=lambda x: x[1]
)
print(f"\n✓ Best Model: {best_model[0]} Regression (MAE: ${best_model[1]:,.4f})")

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[STEP 7] Feature Importance Analysis (Lasso)")
print("=" * 80)

# Create a DataFrame to view coefficients
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
})

# Sort by absolute value to see most impactful features
importance['Abs_Coeff'] = importance['Coefficient'].abs()

# Count features with zero coefficients (feature selection by Lasso)
zero_features = (importance['Coefficient'] == 0).sum()
print(f"\n  Lasso Feature Selection:")
print(f"  → Features with non-zero coefficients: {len(importance) - zero_features}")
print(f"  → Features eliminated (coefficient = 0): {zero_features}")

print("\n  Top 10 Most Important Features (by absolute coefficient):")
print("-" * 80)
top_features = importance.sort_values(by='Abs_Coeff', ascending=False).head(10)
for idx, row in top_features.iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"  {direction} {row['Feature']:<30} Coefficient: {row['Coefficient']:>10.4f}")

# ============================================================================
# 8. VISUALIZATION
# ============================================================================
print("\n[STEP 8] Generating Visualizations...")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Store Sales Forecasting - Model Analysis', fontsize=16, fontweight='bold')

# Subplot 1: Model Comparison (MAE)
ax1 = axes[0, 0]
models = ['Linear', 'Ridge', 'Lasso']
mae_values = [lr_mae, ridge_mae, lasso_mae]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax1.bar(models, mae_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'${val:,.2f}', ha='center', va='bottom', fontsize=10)

# Subplot 2: Top 10 Features (Lasso)
ax2 = axes[0, 1]
top_10 = importance.sort_values(by='Abs_Coeff', ascending=False).head(10)
colors_features = ['#e74c3c' if c < 0 else '#2ecc71' for c in top_10['Coefficient']]
ax2.barh(range(len(top_10)), top_10['Abs_Coeff'], color=colors_features, alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(top_10)))
ax2.set_yticklabels([f[:25] for f in top_10['Feature']], fontsize=9)
ax2.set_xlabel('Absolute Coefficient Value', fontsize=11, fontweight='bold')
ax2.set_title('Top 10 Features (Lasso)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# Subplot 3: Predicted vs Actual (Lasso)
ax3 = axes[1, 0]
# Sample 1000 points for visualization (to avoid overcrowding)
sample_size = min(1000, len(y_test))
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
ax3.scatter(y_test.iloc[sample_indices], lasso_pred[sample_indices], 
            alpha=0.5, s=10, c='#3498db', edgecolors='none')
# Add perfect prediction line
min_val = min(y_test.min(), lasso_pred.min())
max_val = max(y_test.max(), lasso_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Sales', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted Sales', fontsize=11, fontweight='bold')
ax3.set_title('Predicted vs Actual Sales (Lasso)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Subplot 4: Residuals Distribution (Lasso)
ax4 = axes[1, 1]
residuals = y_test - lasso_pred
ax4.hist(residuals, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax4.set_xlabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Residuals Distribution (Lasso)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('sales_forecasting_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Visualization saved as 'sales_forecasting_analysis.png'")

# Show the plot
plt.show()

# ============================================================================
# 9. SUMMARY & INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT SUMMARY & INSIGHTS")
print("=" * 80)

print("\n📊 Key Findings:")
print("-" * 80)

# Performance comparison
if ridge_mae < lr_mae:
    improvement = ((lr_mae - ridge_mae) / lr_mae) * 100
    print(f"  • Ridge regression improved over baseline by {improvement:.2f}%")
else:
    print("  • Ridge regression did not improve over baseline")

if lasso_mae < lr_mae:
    improvement = ((lr_mae - lasso_mae) / lr_mae) * 100
    print(f"  • Lasso regression improved over baseline by {improvement:.2f}%")
else:
    print("  • Lasso regression did not improve over baseline")

# Feature selection insight
print(f"\n  • Lasso identified {len(importance) - zero_features} relevant features")
print(f"  • {zero_features} features were eliminated by Lasso (coefficient = 0)")

# Top driver
top_driver = importance.loc[importance['Abs_Coeff'].idxmax()]
print(f"\n  • Top sales driver: {top_driver['Feature']}")
print(f"    Coefficient: {top_driver['Coefficient']:.4f}")

print("\n✓ Analysis complete!")
print("=" * 80)
