import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import joblib

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve,
    auc, precision_recall_curve, average_precision_score
)

# 📂 Load Dataset
df = pd.read_csv("/Users/nupurshivani/Documents/Heart_Disease_MLProject/Heart Attack.csv") 
print(df.head())
print(df.info())
print(df.describe())

# 📊 Target Class Distribution
sns.countplot(x='class', data=df)
plt.title("Target Class Distribution")
plt.show()

# 📈 Visualize Numerical Columns
num_cols = ['age', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']

# Histograms
df[num_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()

# Boxplots for Outliers
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Boxplots vs Target
for col in num_cols:
    sns.boxplot(x='class', y=col, data=df)
    plt.title(f"{col} vs Class")
    plt.show()

# 🔍 Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 🔍 Gender vs Class
sns.countplot(x='gender', hue='class', data=df)
plt.title("Gender vs Class")
plt.show()

# 🔎 Missing/Duplicate Checks
print(df.isnull().sum())
print("Duplicated Rows:", df.duplicated().sum())

# 🧠 Feature Engineering
df['pulse_pressure'] = df['pressurehight'] - df['pressurelow']
df['bp_ratio'] = df['pressurehight'] / df['pressurelow']
df['high_glucose_flag'] = (df['glucose'] > 140).astype(int)
df['high_troponin'] = (df['troponin'] > 0.4).astype(int)
df['high_kcm'] = (df['kcm'] > 6.0).astype(int)
df['glucose_troponin'] = df['glucose'] * df['troponin']
df['bp_glucose_ratio'] = df['pressurehight'] / (df['glucose'] + 1)

# 🧪 Log Transformation
for col in ['glucose', 'impluse', 'kcm', 'troponin']:
    df[f'{col}_log'] = np.log1p(df[col])

# 🎯 Encode Target
df['class_encoded'] = df['class'].map({'negative': 0, 'positive': 1})

# 🔁 Scale Important Features
scaler = StandardScaler()
scale_cols = ['pressurehight', 'pressurelow', 'pulse_pressure', 'bp_ratio', 'glucose_troponin', 'bp_glucose_ratio']
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# 🧹 Drop Unwanted Columns
drop_cols = ['class', 'glucose', 'impluse', 'kcm', 'troponin', 'age']
df_model = df.drop(columns=drop_cols)

# 🧠 Split Features & Target
X = df_model.drop('class_encoded', axis=1)
y = df_model['class_encoded']

# 🧪 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 🧠 ML Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

# 📊 Cross-Validation Accuracy
print("\n--- Cross-Validation Accuracies ---")
for name, model in models.items():
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: {np.mean(acc):.4f} (+/- {np.std(acc):.4f})")

# 📈 Model Evaluation
print("\n--- Model Test Performance ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("-" * 40)

# 🔍 ROC Curve and Optimal Thresholds
print("\n--- ROC and Optimal Thresholds ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    J = tpr - fpr
    optimal_threshold = thresholds[np.argmax(J)]
    print(f"{name} Optimal Threshold: {optimal_threshold:.3f}")

    y_pred_opt = (y_probs >= optimal_threshold).astype(int)
    print(classification_report(y_test, y_pred_opt))
    print(confusion_matrix(y_test, y_pred_opt))
    print("-" * 50)

# 🔧 Hyperparameter Tuning (XGBoost)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1]
}

search = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_distributions=param_dist, n_iter=30,
    scoring='roc_auc', cv=5, n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)
print(f"\n🔍 Best XGBoost Parameters: {search.best_params_}")
print(f"Best ROC AUC: {search.best_score_:.4f}")

# ✅ Final Model & ROC Curve
model_final = XGBClassifier(**search.best_params_, use_label_encoder=False, eval_metric='logloss', random_state=42)
model_final.fit(X_train, y_train)
y_probs = model_final.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# 🎯 Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
avg_prec = average_precision_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AP = {avg_prec:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()

# 📌 Feature Importance (XGBoost)
importance = model_final.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
features = X_test.columns

plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance[sorted_idx])
plt.xticks(range(len(importance)), features[sorted_idx], rotation=45, ha='right')
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()
# 📌 Save Final Model
joblib.dump(model_final, 'xgboost_heart_attack_model.pkl')
print("Model saved as 'xgboost_heart_attack_model.pkl'")
# 📌 Load Final Model
model_final_loaded = joblib.load('xgboost_heart_attack_model.pkl')
print("Model loaded successfully.")
