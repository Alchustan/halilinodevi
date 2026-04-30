import kagglehub
import pandas as pd
import os
from datetime import timedelta

# --- PREVIOUS STEPS: LOAD, CLEAN & RFM ENGINEERING ---
path = kagglehub.dataset_download("ulrikthygepedersen/online-retail-dataset")
data_file = os.path.join(path, "online_retail.csv")
df = pd.read_csv(data_file, encoding='ISO-8859-1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
df['InvoiceNo'] = df['InvoiceNo'].astype(str)

max_date = df['InvoiceDate'].max()
reference_date = max_date + timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (reference_date - date.max()).days,
    'InvoiceNo': lambda num: num.nunique(),
    'TotalPrice': lambda price: price.sum()
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm = rfm.reset_index()

print("--- RFM Table Loaded ---")

# --- STEP 1 & 2: CREATE R, F, M SCORES ---

print("\n" + "="*30)
print("STEP 2: CREATE R, F, M SCORES")
print("="*30)

# Recency Score: Lower recency is better, so labels are reversed [5, 4, 3, 2, 1]
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

# Frequency Score: Higher frequency is better. 
# Using rank(method='first') to handle duplicate bin edges (many customers have only 1-2 purchases)
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

# Monetary Score: Higher monetary is better.
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Convert scores to integers for calculation
rfm['R_score'] = rfm['R_score'].astype(int)
rfm['F_score'] = rfm['F_score'].astype(int)
rfm['M_score'] = rfm['M_score'].astype(int)

print("R, F, M scores created successfully.")

# --- STEP 3: CREATE COMBINED RFM SCORE ---

print("\n" + "="*30)
print("STEP 3: CREATE COMBINED RFM SCORE")
print("="*30)

# Numerical sum for simple sorting and overall value assessment
rfm['RFM_SUM'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# String representation (e.g., "555") for specific segment identification
rfm['RFM_STRING'] = (rfm['R_score'].astype(str) + 
                    rfm['F_score'].astype(str) + 
                    rfm['M_score'].astype(str))

print("Combined RFM scores (Sum and String) calculated.")
print(rfm[['CustomerID', 'R_score', 'F_score', 'M_score', 'RFM_SUM', 'RFM_STRING']].head())

# --- STEP 4: ANALYZE SCORE DISTRIBUTION ---

print("\n" + "="*30)
print("STEP 4: ANALYZE SCORE DISTRIBUTION")
print("="*30)

print("\nValue Counts for Individual Scores:")
for col in ['R_score', 'F_score', 'M_score']:
    print(f"\n{col} distribution:")
    print(rfm[col].value_counts().sort_index())

print("\nRFM_SUM Distribution (Total counts per total score):")
print(rfm['RFM_SUM'].value_counts().sort_index())

# --- STEP 5: INSPECT HIGH AND LOW VALUE CUSTOMERS ---

print("\n" + "="*30)
print("STEP 5: INSPECT HIGH AND LOW VALUE CUSTOMERS")
print("="*30)

# Top Customers: Those with the highest possible score
# These represent "Champions" who are recent, frequent, and high-spending.
top_customers = rfm[rfm['RFM_SUM'] == 15].sort_values('Monetary', ascending=False)
print("\nTop Customers (RFM Sum = 15 - 'Champions'):")
print(top_customers.head(10))

# Bottom Customers: Those with the lowest possible score
# These represent "Lost" customers who haven't bought recently, rarely buy, and spend little.
bottom_customers = rfm[rfm['RFM_SUM'] == 3].sort_values('Monetary', ascending=True)
print("\nLowest Value Customers (RFM Sum = 3 - 'Hibernating/Lost'):")
print(bottom_customers.head(10))

print("\nRFM Scoring Completed Successfully.")
