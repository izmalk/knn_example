import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

start = time.time()

# generate random dataset
X, y = make_blobs(n_samples=5000, n_features=2, centers=5, cluster_std=1.5, random_state=4)
print(f"Dataset generated. Time spent: {time.time() - start}")

# visualize the dataset
# plt.style.use('seaborn')

print("Generating dataset visualisation...")
plt.figure(1, figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='*', s=100, edgecolors='black')
plt.title("Initial dataset visualization", fontsize=20)
plt.show()

start = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(f"Dataset split complete. Time spent: {time.time() - start}")


print(f"Starting to train models with K=1..10")
# Explore K=1..10
k = [None] * 10
y_predicted = [None] * 10
acc = [0.01] * 10
max_acc = 0
for n in range(10):
    start = time.time()
    k[n] = KNeighborsClassifier(n_neighbors=n + 1)
    k[n].fit(X_train, y_train)  # Training the model
    y_predicted[n] = k[n].predict(X_test)  # Predicting values with the model
    acc[n] = accuracy_score(y_test, y_predicted[n]) * 100  # Assessing accuracy of the model
    if max_acc < acc[n]:
        max_acc = acc[n]
        max_acc_k = n
    print(f"Accuracy with k={n}: {acc[n]}. Time spent: {time.time() - start}")

print("Generating accuracy visualisation...")
plt.figure(2, figsize=(5, 5))
plt.title('Accuracy for K=1..10')
x = [None] * 10
for n in range(10):
    x[n] = n+1
plt.plot(x, acc, color='red', marker='x', linestyle='solid', linewidth=1)
plt.grid(True)
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.show()

print("Generating prediction visualisation...")
plt.figure(3, figsize=(5, 5))
# plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_predicted[max_acc_k], marker='*', s=100, edgecolors='black')
plt.title(f"Predicted values with k={max_acc_k+1}", fontsize=20)
plt.show()

"""
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_predicted[1], marker='*', s=100, edgecolors='black')
plt.title("Predicted values with k=1", fontsize=20)
plt.show()
"""

"""
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_5, marker= '*', s=100,edgecolors='black')
plt.title("Predicted values with k=5", fontsize=20)
"""
