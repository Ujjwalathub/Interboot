import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- 1. Load Data ---
print("Loading datasets...")
train_df = pd.read_csv(r"E:\Project\Data\train.csv")
oil_df = pd.read_csv(r"E:\Project\Data\oil.csv")

# --- 2. Preprocessing & Merging ---
# Convert 'date' to datetime objects for accurate merging
train_df['date'] = pd.to_datetime(train_df['date'])
oil_df['date'] = pd.to_datetime(oil_df['date'])

# Merge external data onto the training data
# We use 'left' join to preserve all rows from the sales dataset
merged_df = pd.merge(train_df, oil_df, on='date', how='left')

print(f"Data shape before merge: {train_df.shape}")
print(f"Data shape after merge: {merged_df.shape}")

# --- 3. Handle Missing Values (Imputation) ---
# Check missing values in the new 'dcoilwtico' (oil price) column
missing_oil = merged_df['dcoilwtico'].isnull().sum()
print(f"Missing oil prices before cleaning: {missing_oil}")

# Strategy: Forward fill (use Friday's price for Sat/Sun)
# Then Backward fill (just in case the very first date is missing)
merged_df['dcoilwtico'] = merged_df['dcoilwtico'].ffill().bfill()

# --- 4. Feature Engineering (Same as Task 1) ---
# We still need to process dates and categories for the model to work
merged_df['year'] = merged_df['date'].dt.year
merged_df['month'] = merged_df['date'].dt.month
merged_df['day_of_week'] = merged_df['date'].dt.dayofweek

# Drop non-numeric columns not needed for training
# keeping 'dcoilwtico' (oil) as our new feature!
X_raw = merged_df.drop(['sales', 'date', 'id'], axis=1)
y = merged_df['sales']

# One-Hot Encoding for 'family'
X = pd.get_dummies(X_raw, columns=['family'], drop_first=True)

# Check correlations: Does oil price actually correlate with sales?
oil_corr = merged_df[['sales', 'dcoilwtico']].corr().iloc[0,1]
print(f"\nCorrelation between Oil Price and Sales: {oil_corr:.4f}")

# --- 5. Train & Evaluate with External Data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Ridge Regression (usually better than simple Linear for many features)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# --- 6. Results ---
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\n--- Model Performance with External Data (Oil) ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Compare this MAE to your Task 1 result. 
# Did it go down? If yes, the external data helped!

# --- 7. Visualization of Impact ---
# Visualize Oil Price vs Sales over time
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Average Sales
daily_sales = merged_df.groupby('date')['sales'].mean()
ax1.plot(daily_sales.index, daily_sales, color='blue', label='Avg Sales')
ax1.set_ylabel('Average Sales', color='blue')

# Create a second y-axis for Oil Price
ax2 = ax1.twinx()
daily_oil = merged_df.groupby('date')['dcoilwtico'].mean()
ax2.plot(daily_oil.index, daily_oil, color='red', linestyle='--', label='Oil Price')
ax2.set_ylabel('Oil Price ($)', color='red')

plt.title('Sales vs. Oil Price Trends')
plt.show()
