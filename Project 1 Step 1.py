#Step 1
import pandas as pd

df = pd.read_csv("Project 1 Data")

print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())
