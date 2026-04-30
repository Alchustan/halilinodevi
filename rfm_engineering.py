import kagglehub
import pandas as pd
import os
from datetime import timedelta

# --- PREVIOUS STEPS: LOAD & CLEANING ---
path = kagglehub.dataset_download("ulrikthygepedersen/online-retail-dataset")
data_file = os.path.join(path, "online_retail.csv")
df = pd.read_csv(data_file, encoding='ISO-8859-1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
df['InvoiceNo'] = df['InvoiceNo'].astype(str)

print("--- Data Loaded and Cleaned ---")

# --- STEP 1: DEFINE REFERENCE DATE ---

print("\n" + "="*30)
print("STEP 1: DEFINE REFERENCE DATE")
print("="*30)

max_date = df['InvoiceDate'].max()
# Reference date is set to one day after the last transaction in the dataset
reference_date = max_date + timedelta(days=1)

print(f"Dataset Max Date: {max_date}")
print(f"Reference Date:   {reference_date}")

# --- STEP 2 & 3: COMPUTE RFM METRICS & CREATE RFM TABLE ---

print("\n" + "="*30)
print("STEP 2 & 3: COMPUTE RFM METRICS")
print("="*30)

# Grouping by CustomerID and calculating Recency, Frequency, and Monetary
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (reference_date - date.max()).days, # Recency
    'InvoiceNo': lambda num: num.nunique(),                        # Frequency
    'TotalPrice': lambda price: price.sum()                        # Monetary
})

# Renaming columns for clarity
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Resetting index to have CustomerID as a column
rfm = rfm.reset_index()

print("RFM Table successfully created.")

# --- STEP 4: VALIDATE RFM VALUES ---

print("\n" + "="*30)
print("STEP 4: VALIDATE RFM VALUES")
print("="*30)

print("\nFirst 5 rows of RFM table:")
print(rfm.head())

print("\nSummary Statistics of RFM Metrics:")
print(rfm.describe())

# Check for anomalies
neg_recency = (rfm['Recency'] < 0).sum()
zero_freq = (rfm['Frequency'] <= 0).sum()
zero_monetary = (rfm['Monetary'] <= 0).sum()

print("\nAnomalies Check:")
print(f"Customers with Negative Recency: {neg_recency}")
print(f"Customers with Zero Frequency:   {zero_freq}")
print(f"Customers with Zero Monetary:    {zero_monetary}")

# --- STEP 5: SORT AND INSPECT CUSTOMERS ---

print("\n" + "="*30)
print("STEP 5: SORT AND INSPECT CUSTOMERS")
print("="*30)

print("\nTop 10 High Spenders (Highest Monetary):")
print(rfm.sort_values(by='Monetary', ascending=False).head(10))

print("\nTop 10 Most Recent Buyers (Lowest Recency):")
print(rfm.sort_values(by='Recency', ascending=True).head(10))

print("\nRFM Feature Engineering Completed Successfully.")
