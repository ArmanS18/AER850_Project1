import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Step 1

file_path = r'C:\Users\arman\OneDrive\Documents\GitHub\AER850_Project1\Project 1 Data.csv'
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

#Step 2

#Data Inspection
print ("Summary statistics:")
print(df.describe())

features = ['X', 'Y', 'Z']
target = 'Step'

#Histogram
df[features].hist(figsize=(10, 6), bins=20)
plt.suptitle("Fetaure Distribution", fontsize=14)
plt.show()

#Boxplots
plt.figure(figsize=(12, 6))
for i, col in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=target, y=col, data=df)
    plt.title(f"{col} by {target}")
plt.tight_layout()
plt.show()

#Pairplot
sns.pairplot(df, hue=target, vars=features, palette="Set2")
plt.show()

#Class Balance

plt.figure(figsize=(6, 4))
df[target].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Step")
plt.ylabel("Count")
plt.show()
