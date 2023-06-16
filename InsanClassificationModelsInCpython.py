import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Download the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True)

# Split the dataset into training and testing sets
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Clean the data by removing missing values and invalid data
def clean_data(X, y):
    idx = np.isnan(X).any(axis=1)
    X = X[~idx]
    y = y[~idx]
    return X, y

X_train, y_train = clean_data(X_train, y_train)
X_test, y_test = clean_data(X_test, y_test)

# Find the best value for the hyperparameter k using GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9]}
knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1, verbose=2)
knn_clf.fit(X_train, y_train)

# Determine the best value for the hyperparameter k
best_k = knn_clf.best_params_['n_neighbors']
print(f"Best k: {best_k}")

# Train the k-Nearest Neighbors algorithm using the best value for k and PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn_clf = KNeighborsClassifier(n_neighbors=best_k)
knn_clf.fit(X_train_pca, y_train)

# Test the algorithm on the testing set
y_pred = knn_clf.predict(X_test_pca)

# Calculate the accuracy of the classification
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (kNN): {accuracy}")

# Train a Random Forest classifier and calculate its accuracy
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Random Forest): {accuracy}")

# Train a MLP classifier and calculate its accuracy
mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp_clf.fit(X_train, y_train)
y_pred = mlp_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (MLP): {accuracy}")

# Display the misclassified images for the kNN classifier
wrong_images = X_test[y_test != y_pred]
wrong_labels = y_pred[y_test != y_pred]
correct_labels = y_test[y_test != y_pred]

plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(np.reshape(wrong_images[i], (28, 28)), cmap=plt.cm.gray)
    plt.title(f"Predicted: {wrong_labels[i]}, Actual: {correct_labels[i]}")
    plt.axis('off')
plt.show()
