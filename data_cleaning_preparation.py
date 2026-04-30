import kagglehub
import pandas as pd
import os

# --- PREVIOUS STEPS: LOAD & INITIAL EDA ---

# Download and load the dataset (reusing logic from previous step)
path = kagglehub.dataset_download("ulrikthygepedersen/online-retail-dataset")
data_file = os.path.join(path, "online_retail.csv")

# Load with correct encoding
df = pd.read_csv(data_file, encoding='ISO-8859-1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("--- Data Loaded Successfully ---")
print(f"Initial Shape: {df.shape}")

# --- STEP 1: HANDLE MISSING VALUES ---

print("\n" + "="*30)
print("STEP 1: HANDLE MISSING VALUES")
print("="*30)

initial_rows = len(df)
# Identify and remove rows where CustomerID is missing
df = df.dropna(subset=['CustomerID'])
removed_rows_na = initial_rows - len(df)

print(f"Rows removed (missing CustomerID): {removed_rows_na}")
print(f"Current Shape: {df.shape}")

# --- STEP 2: REMOVE INVALID TRANSACTIONS ---

print("\n" + "="*30)
print("STEP 2: REMOVE INVALID TRANSACTIONS")
print("="*30)

# Count rows with Quantity <= 0
invalid_quantity = len(df[df['Quantity'] <= 0])
# Count rows with UnitPrice <= 0
invalid_price = len(df[df['UnitPrice'] <= 0])

# Filter the dataframe
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

print(f"Rows removed (Quantity <= 0): {invalid_quantity}")
print(f"Rows removed (UnitPrice <= 0): {invalid_price}")
print(f"Current Shape: {df.shape}")

# --- STEP 3: CREATE NEW FEATURE ---

print("\n" + "="*30)
print("STEP 3: CREATE NEW FEATURE")
print("="*30)

# TotalPrice calculation
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("Verified 'TotalPrice' column (Sample rows):")
print(df[['InvoiceNo', 'Quantity', 'UnitPrice', 'TotalPrice']].head())

# --- STEP 4: DATA TYPE FIXES ---

print("\n" + "="*30)
print("STEP 4: DATA TYPE FIXES")
print("="*30)

# CustomerID to string: Important because CustomerID is a categorical identifier. 
# Treating it as float/numeric can lead to misleading statistical calculations 
# and issues like trailing zeros (12345.0).
df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

# InvoiceNo to string: Essential because InvoiceNo can contain non-numeric characters 
# (e.g., 'C' for credit notes), and as an identifier, it should be treated as text.
df['InvoiceNo'] = df['InvoiceNo'].astype(str)

print("Data types after fixes:")
print(df[['CustomerID', 'InvoiceNo']].dtypes)
print("\nSample IDs (ensuring no trailing .0):")
print(df[['CustomerID', 'InvoiceNo']].head())

# --- STEP 5: FINAL SANITY CHECK ---

print("\n" + "="*30)
print("STEP 5: FINAL SANITY CHECK")
print("="*30)

print(f"Final dataset shape: {df.shape}")

print("\nRemaining missing values per column:")
print(df.isnull().sum())

print("\nBasic statistics of 'TotalPrice':")
print(df['TotalPrice'].describe())

print("\nData Preparation Completed Successfully.")
