import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import time

random_state = 42

print("Loading dataset...")
df = pd.read_csv('Student_performance_data _.csv')
print(f"Dataset shape: {df.shape}")

df = df.dropna()
print(f"Dataset shape after removing NaN: {df.shape}")

y = df['GradeClass'].astype(int)
X = df.drop(['StudentID', 'GradeClass'], axis=1)

print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(y.unique())}")
print(f"Class distribution: {dict(Counter(y))}")

target_counts = Counter(y)
plt.figure(figsize=(10, 5))
plt.bar(target_counts.keys(), target_counts.values(), color='darkgreen')
plt.xticks(sorted(target_counts.keys()))
plt.xlabel("GradeClass (Category)")
plt.ylabel("Count")
plt.title("Distribution of Student Grade Classes")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

classifiers = {
    'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
    'Support Vector Machine (SVM)': SVC(random_state=random_state),
    'Decision Tree (DT)': DecisionTreeClassifier(random_state=random_state),
    'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': MultinomialNB()
}

# Add XGBoost and LightGBM if available
if XGBOOST_AVAILABLE:
    classifiers['XGBoost'] = xgb.XGBClassifier(
        random_state=random_state,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss'
    )

if LIGHTGBM_AVAILABLE:
    classifiers['LightGBM'] = lgb.LGBMClassifier(
        random_state=random_state,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1  # Suppress output
    )

results = []

print("\n" + "="*80)
print("TRAINING AND EVALUATING MODELS")
print("="*80)

for name, classifier in classifiers.items():
    print(f"\nTraining {name}...")
    
    if name == 'Naive Bayes':
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', classifier)
        ])
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        start_time = time.time()
        y_pred = pipeline.predict(X_test)
        prediction_time = time.time() - start_time
    elif name in ['XGBoost', 'LightGBM']:
        # Fit and predict directly for XGBoost and LightGBM
        start_time = time.time()
        classifier.fit(X_train, y_train)
        training_time = time.time() - start_time
        start_time = time.time()
        y_pred = classifier.predict(X_test)
        prediction_time = time.time() - start_time
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', classifier)
        ])
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        start_time = time.time()
        y_pred = pipeline.predict(X_test)
        prediction_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results.append({
        'Method': name,
        'Dataset': 'Student Performance',
        'Amount of Data': f"{X_train.shape[0]} train, {X_test.shape[0]} test",
        'Speed (Training)': f"{training_time:.4f}s",
        'Speed (Prediction)': f"{prediction_time:.4f}s",
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1-Score': f"{f1:.4f}"
    })

    print(f"Training time: {training_time:.4f}s")
    print(f"Prediction time: {prediction_time:.4f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("COMPARISON TABLE")
print("="*80)
print(results_df.to_string(index=False))

results_df.to_csv('multiclass_results.csv', index=False)
print(f"\nResults saved to 'multiclass_results.csv'")

plt.figure(figsize=(12, 6))
methods = results_df['Method']
accuracies = [float(acc) for acc in results_df['Accuracy']]

plt.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title('Accuracy Comparison of Different Classification Methods', fontsize=14, fontweight='bold')
plt.xlabel('Classification Method', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

best_model_idx = np.argmax(accuracies)
best_model = methods.iloc[best_model_idx]
best_accuracy = accuracies[best_model_idx]

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Best performing model: {best_model}")
print(f"Best accuracy: {best_accuracy:.4f}")
print(f"Total models tested: {len(classifiers)}")
print(f"Dataset: Student Performance Data")
print(f"Total samples: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(y.unique())}")