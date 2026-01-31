# save_model.py
import pandas as pd
import joblib
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 1. Load and Prepare Data (Same logic as Task 1/2)
print("Loading and processing data...")
df = pd.read_csv(r"E:\Project\Data\train.csv")
oil = pd.read_csv(r"E:\Project\Data\oil.csv")

# Merge Oil Data
df['date'] = pd.to_datetime(df['date'])
oil['date'] = pd.to_datetime(oil['date'])
df = pd.merge(df, oil, on='date', how='left')
df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()

# Feature Engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek

# One-Hot Encoding
# We drop 'id' and 'date'. We keep 'sales' as target.
X_raw = df.drop(['id', 'date', 'sales'], axis=1)
y = df['sales']
X = pd.get_dummies(X_raw, columns=['family'], drop_first=True)

# 2. Scale the Data
print("Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train the Model (Lasso - as it was your best performer)
print("Training Lasso model...")
model = Lasso(alpha=0.1)
model.fit(X_scaled, y)

# 4. Save Artifacts for the App
print("Saving model artifacts...")
joblib.dump(model, 'final_sales_model.pkl')    # The Brain
joblib.dump(scaler, 'final_scaler.pkl')        # The Translator (Scaling)
joblib.dump(X.columns, 'model_columns.pkl')    # The Map (Column Names)

print("Done! Files saved: 'final_sales_model.pkl', 'final_scaler.pkl', 'model_columns.pkl'")
