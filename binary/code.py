import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"Warning: XGBoost not available. Error: {e}")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("Using custom LightGBM implementation")
except (ImportError, Exception) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"Warning: LightGBM not available. Error: {e}")
import matplotlib.pyplot as plt
from collections import Counter
import time

random_state = 42

df = pd.read_csv('Surgical-deepnet.csv')

y = df['complication']
X = df.drop('complication', axis=1)


target_counts = Counter(y)
labels = [f"No Complication ({target_counts[0]})", f"Complication ({target_counts[1]})"]
counts = [target_counts[0], target_counts[1]]

plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['skyblue', 'salmon'])
plt.ylabel("Count")
plt.title("Distribution of Surgical Complications")
plt.tight_layout()
plt.savefig('complication_distribution.png')
plt.close()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lr = LogisticRegression(random_state=random_state, max_iter=1000)
DT = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=random_state)
SV = SVC(kernel='linear', C=1.0, random_state=random_state)
rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state)

# Create XGBoost and LightGBM models only if available
if XGBOOST_AVAILABLE:
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss')
else:
    xgb_model = None

if LIGHTGBM_AVAILABLE:
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1)
else:
    lgb_model = None

lr.fit(x_train, y_train)
DT.fit(x_train, y_train)
SV.fit(x_train_scaled, y_train)  # SVM benefits from scaling
rf.fit(x_train, y_train)
knn.fit(x_train_scaled, y_train)  # KNN benefits from scaling
nb.fit(x_train, y_train)
gb.fit(x_train, y_train)
mlp.fit(x_train_scaled, y_train)  # Neural network benefits from scaling

# Train XGBoost and LightGBM only if available
if XGBOOST_AVAILABLE and xgb_model is not None:
    xgb_model.fit(x_train, y_train)

if LIGHTGBM_AVAILABLE and lgb_model is not None:
    lgb_model.fit(x_train, y_train)

def measure_training_time(model, X_train, y_train):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    return end_time - start_time

total_samples = len(df)
train_samples = len(x_train)
test_samples = len(x_test)

lr_time = measure_training_time(LogisticRegression(random_state=random_state, max_iter=1000), x_train, y_train)
dt_time = measure_training_time(DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=random_state), x_train, y_train)
svm_time = measure_training_time(SVC(kernel='linear', C=1.0, random_state=random_state), x_train_scaled, y_train)
rf_time = measure_training_time(RandomForestClassifier(n_estimators=100, random_state=random_state), x_train, y_train)
knn_time = measure_training_time(KNeighborsClassifier(n_neighbors=5), x_train_scaled, y_train)
nb_time = measure_training_time(GaussianNB(), x_train, y_train)
gb_time = measure_training_time(GradientBoostingClassifier(n_estimators=100, random_state=random_state), x_train, y_train)
mlp_time = measure_training_time(MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state), x_train_scaled, y_train)

# Measure XGBoost and LightGBM training times only if available
if XGBOOST_AVAILABLE:
    xgb_time = measure_training_time(xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss'), x_train, y_train)
else:
    xgb_time = 0.0

if LIGHTGBM_AVAILABLE:
    lgb_time = measure_training_time(lgb.LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1), x_train, y_train)
else:
    lgb_time = 0.0

lr_accuracy = lr.score(x_test, y_test)
dt_accuracy = DT.score(x_test, y_test)
svm_accuracy = SV.score(x_test_scaled, y_test)
rf_accuracy = rf.score(x_test, y_test)
knn_accuracy = knn.score(x_test_scaled, y_test)
nb_accuracy = nb.score(x_test, y_test)
gb_accuracy = gb.score(x_test, y_test)
mlp_accuracy = mlp.score(x_test_scaled, y_test)

# Get XGBoost and LightGBM accuracies only if available
if XGBOOST_AVAILABLE and xgb_model is not None:
    xgb_accuracy = xgb_model.score(x_test, y_test)
else:
    xgb_accuracy = 0.0

if LIGHTGBM_AVAILABLE and lgb_model is not None:
    lgb_accuracy = lgb_model.score(x_test, y_test)
else:
    lgb_accuracy = 0.0

# Build results data dynamically based on available models
methods = [
    'Logistic Regression', 
    'Support Vector Machine (SVM)', 
    'Decision Tree (DT)',
    'Random Forest (RF)',
    'K-Nearest Neighbors (KNN)',
    'Naive Bayes (NB)',
    'Gradient Boosting (GB)',
    'Neural Network (MLP)'
]

speeds = [lr_time, svm_time, dt_time, rf_time, knn_time, nb_time, gb_time, mlp_time]
accuracies = [lr_accuracy, svm_accuracy, dt_accuracy, rf_accuracy, knn_accuracy, nb_accuracy, gb_accuracy, mlp_accuracy]

# Add XGBoost and LightGBM if available
if XGBOOST_AVAILABLE:
    methods.append('XGBoost (XGB)')
    speeds.append(xgb_time)
    accuracies.append(xgb_accuracy)

if LIGHTGBM_AVAILABLE:
    methods.append('LightGBM (LGB)')
    speeds.append(lgb_time)
    accuracies.append(lgb_accuracy)

results_data = {
    'Method': methods,
    'Dataset': ['Surgical-deepnet.csv'] * len(methods),
    'Amount of Data': [f'{total_samples} samples'] * len(methods),
    'Speed (seconds)': [f'{speed:.4f}' for speed in speeds],
    'Accuracy': [f'{accuracy:.4f}' for accuracy in accuracies]
}

results_df = pd.DataFrame(results_data)

print("\n" + "="*80)
print("MACHINE LEARNING MODELS COMPARISON TABLE")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)
print(f"Dataset split: {train_samples} training samples, {test_samples} test samples")
print("="*80)