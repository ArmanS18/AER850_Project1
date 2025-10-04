import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Step 1------------------------------------------------------------------------

file_path = r'C:\Users\arman\OneDrive\Documents\GitHub\AER850_Project1\Project 1 Data.csv'
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())

df = df.dropna().reset_index(drop=True)
print("\nMissing values:")
print(df.isnull().sum())
print("\mData shape after dropping missing values:", df.shape)

#Step 2------------------------------------------------------------------------

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
    sns.boxplot(x=target, y=col, data=df, palette="pastel")
    plt.title(f"{col} by {target}")
plt.tight_layout()
plt.show()

#Pairplot
sns.pairplot(df, hue=target, vars=features, palette="Set2", diag_kind='hist')
plt.suptitle("Pairwise Feature Relationship by Step", y=1.02)
plt.show()

#Class Balance

plt.figure(figsize=(6, 4))
df[target].value_counts().plot(kind='bar')
plt.title("Class Distribution (Maintenance Step)")
plt.xlabel("Step")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Step 3------------------------------------------------------------------------

#Feature Correlation

corr_matrix = df[features].corr(method='pearson')
print("Feature Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Features")
plt.show()

#Correlation with Target

if not np.issubdtype(df[target].dtype, np.number):
    df[target] = df[target].astype('category').cat.codes
    
corr_with_target = df[features + [target]].corr(method='pearson')
print("\nCorrelation with Target (Step:")
print(corr_with_target[target])

plt.figure(figsize=(6, 4))
sns.heatmap(corr_with_target, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation of Features with Targt (Step)")
plt.show()

#Interpretation
print("\nInterpretation Guide:")
print("- High positive correlation (close to +1): Fetaures move together.")
print("- High negative correlation (close to -1): One increases as the other decreases.")
print("- Near 0: Features are weakly related.")
print("- Stronger correaltion with 'step' means that t=feature is more predictive.")
