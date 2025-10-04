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

# Standardize features for algorithms that benefit from scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create all machine learning models
lr = LogisticRegression(random_state=random_state, max_iter=1000)
DT = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=random_state)
SV = SVC(kernel='linear', C=1.0, random_state=random_state)
rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state)

# Train all models
lr.fit(x_train, y_train)
DT.fit(x_train, y_train)
SV.fit(x_train_scaled, y_train)  # SVM benefits from scaling
rf.fit(x_train, y_train)
knn.fit(x_train_scaled, y_train)  # KNN benefits from scaling
nb.fit(x_train, y_train)
gb.fit(x_train, y_train)
mlp.fit(x_train_scaled, y_train)  # Neural network benefits from scaling

def measure_training_time(model, X_train, y_train):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    return end_time - start_time

total_samples = len(df)
train_samples = len(x_train)
test_samples = len(x_test)

# Measure training times for all models
lr_time = measure_training_time(LogisticRegression(random_state=random_state, max_iter=1000), x_train, y_train)
dt_time = measure_training_time(DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=random_state), x_train, y_train)
svm_time = measure_training_time(SVC(kernel='linear', C=1.0, random_state=random_state), x_train_scaled, y_train)
rf_time = measure_training_time(RandomForestClassifier(n_estimators=100, random_state=random_state), x_train, y_train)
knn_time = measure_training_time(KNeighborsClassifier(n_neighbors=5), x_train_scaled, y_train)
nb_time = measure_training_time(GaussianNB(), x_train, y_train)
gb_time = measure_training_time(GradientBoostingClassifier(n_estimators=100, random_state=random_state), x_train, y_train)
mlp_time = measure_training_time(MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state), x_train_scaled, y_train)

# Get accuracies for all models
lr_accuracy = lr.score(x_test, y_test)
dt_accuracy = DT.score(x_test, y_test)
svm_accuracy = SV.score(x_test_scaled, y_test)
rf_accuracy = rf.score(x_test, y_test)
knn_accuracy = knn.score(x_test_scaled, y_test)
nb_accuracy = nb.score(x_test, y_test)
gb_accuracy = gb.score(x_test, y_test)
mlp_accuracy = mlp.score(x_test_scaled, y_test)

results_data = {
    'Method': [
        'Logistic Regression', 
        'Support Vector Machine (SVM)', 
        'Decision Tree (DT)',
        'Random Forest (RF)',
        'K-Nearest Neighbors (KNN)',
        'Naive Bayes (NB)',
        'Gradient Boosting (GB)',
        'Neural Network (MLP)'
    ],
    'Dataset': ['Surgical-deepnet.csv'] * 8,
    'Amount of Data': [f'{total_samples} samples'] * 8,
    'Speed (seconds)': [
        f'{lr_time:.4f}', 
        f'{svm_time:.4f}', 
        f'{dt_time:.4f}',
        f'{rf_time:.4f}',
        f'{knn_time:.4f}',
        f'{nb_time:.4f}',
        f'{gb_time:.4f}',
        f'{mlp_time:.4f}'
    ],
    'Accuracy': [
        f'{lr_accuracy:.4f}', 
        f'{svm_accuracy:.4f}', 
        f'{dt_accuracy:.4f}',
        f'{rf_accuracy:.4f}',
        f'{knn_accuracy:.4f}',
        f'{nb_accuracy:.4f}',
        f'{gb_accuracy:.4f}',
        f'{mlp_accuracy:.4f}'
    ]
}

results_df = pd.DataFrame(results_data)

print("\n" + "="*80)
print("MACHINE LEARNING MODELS COMPARISON TABLE")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)
print(f"Dataset split: {train_samples} training samples, {test_samples} test samples")
print("="*80)