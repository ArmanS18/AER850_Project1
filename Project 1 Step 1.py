import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")

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

#Step 4------------------------------------------------------------------------

#Data split into Train and Test

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

#Model 1: Support Vector Machine
svm_pipeline = Pipeline([
    ('scalar', StandardScaler()),
    ('model', SVC())
])

svm_param_grid = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf', 'poly'],
    'model__gamma': ['scale', 'auto']
}

svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=5,
                        scoring='accuracy', n_jobs=-1, verbose=1)
svm_grid.fit(X_train, y_train)

print("\nBest aprameters for SVM:", svm_grid.best_params_)
print("Best CV accuracy for SVM:", round(svm_grid.best_score_, 3))

#Model 2: Random Forest
rf_pipeline = Pipeline([
    ('scalar', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 5, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5,
                       scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

print("\nBest parameters for Random Forest:", rf_grid.best_params_)
print("Best CV accuracy for RF:", round(rf_grid.best_score_, 3))

#Model 3: K-Nearest Neighbours
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

knn_param_grid = {
    'model__n_neighbors': [3, 5, 7, 9],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan']
}

knn_grid = GridSearchCV(knn_pipeline, knn_param_grid, cv=5,
                        scoring='accuracy', n_jobs=-1, verbose=1)
knn_grid.fit(X_train, y_train)

print("\nBest parameters for KNN:", knn_grid.best_params_)
print("Best CV accuracy for KNN:", round(knn_grid.best_score_, 3))

#RandomizedSearchCV Model
rf_random = RandomizedSearchCV(
    rf_pipeline,
    rf_param_grid,
    n_iter=5,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_random.fit(X_train, y_train)
print("\nRandomizedSearchCV best RF params:", rf_random.best_params_)
print("RandomizedSearchCV best RF accuracy:", round(rf_random.best_score_, 3))

#All Model Evaluation
models = {
    "SVM": svm_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_,
    "KNN": knn_grid.best_estimator_,
    "RF Randomized": rf_random.best_estimator_
}

print("\nModel Performance on Test Data:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name:15s} | Accuracy: {acc:.3f} | Precision: {prec:.3f} | F1: {f1:.3f}")

#Step 5------------------------------------------------------------------------

best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test)

print("\n===== SVM Model Evaluation =====")
print("Best Parameters:", svm_grid.best_params_)
print("Test Accuracy:", round(accuracy_score(y_test, y_pred_svm), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

#Decision Tree
dt_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', DecisionTreeClassifier(random_state=42))
])
dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)

#Random Forest
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

#Evaluatoin of both
print("\n===== Decision Tree Evaluation =====")
print("Test Accuracy:", round(accuracy_score(y_test, y_pred_dt), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

print("\n===== Random Forest Evaluation =====")
print("Test Accuracy:", round(accuracy_score(y_test, y_pred_rf), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

#Comparison Summary
print("\n===== Model Accuracy Comparison =====")
print("SVM Accuracy:           ", round(accuracy_score(y_test, y_pred_svm), 3))
print("Decision Tree Accuracy: ", round(accuracy_score(y_test, y_pred_dt), 3))
print("Random Forest Accuracy: ", round(accuracy_score(y_test, y_pred_rf), 3))

#Confusion Matrix Visualisation)
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()