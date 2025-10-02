#Step 1
import pandas as pd

file_path = r'C:\Users\arman\OneDrive\Documents\GitHub\AER850_Project1\Project 1 Data.csv'
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())
