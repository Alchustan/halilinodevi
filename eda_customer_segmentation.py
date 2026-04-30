import kagglehub
import pandas as pd
import os

# --- STEP 1: LOAD DATA ---

# Download the dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("ulrikthygepedersen/online-retail-dataset")
print("Path to dataset files:", path)

# Identify the correct data file inside the downloaded folder
files = os.listdir(path)
print("Files in downloaded path:", files)

# Looking for a CSV or Excel file
data_file = None
for f in files:
    if f.endswith('.csv') or f.endswith('.xlsx'):
        data_file = os.path.join(path, f)
        break

if data_file:
    print(f"Loading data from: {data_file}")
    # Handling potential encoding issues with 'ISO-8859-1' which is common for retail datasets
    try:
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file, encoding='ISO-8859-1')
        else:
            df = pd.read_excel(data_file)
    except Exception as e:
        print(f"Error loading with ISO-8859-1: {e}. Trying default...")
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_excel(data_file)
else:
    print("No CSV or Excel file found in the downloaded folder.")
    exit()

# --- STEP 2: BASIC INSPECTION ---

print("\n" + "="*30)
print("STEP 2: BASIC INSPECTION")
print("="*30)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# --- STEP 3: DATA QUALITY CHECKS ---

print("\n" + "="*30)
print("STEP 3: DATA QUALITY CHECKS")
print("="*30)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSummary statistics for numerical columns:")
print(df.describe())

print("\nUnique values:")
print(f"Unique CustomerIDs: {df['CustomerID'].nunique()}")
print(f"Unique InvoiceNos: {df['InvoiceNo'].nunique()}")

# --- STEP 4: DATE HANDLING ---

print("\n" + "="*30)
print("STEP 4: DATE HANDLING")
print("="*30)

print("Converting 'InvoiceDate' to datetime...")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("\nVerified 'InvoiceDate' dtype:")
print(df['InvoiceDate'].dtype)

print("\nSample 'InvoiceDate' values:")
print(df['InvoiceDate'].head())

print("\nEDA Completed Successfully.")
