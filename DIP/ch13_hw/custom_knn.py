import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class CustomKNN:
    def __init__(self, n_neighbors = 5):
        self.X_train = None
        self.y_train = None
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# 1. Generate a synthetic dataset
x, y = make_classification(
    n_samples = 200, n_features = 2, n_classes = 2, n_informative = 2, n_redundant = 0, random_state = 42
)

# Split the dataset into training and testing sets(70% training, 30% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# 2. Visualize the dataset
plt.figure(figsize = (8, 6))
plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color = 'blue', label = 'Class 0(Train)', alpha = 0.6)
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color = 'red', label = 'Class 1(Train)', alpha = 0.6)
plt.title("Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# 3. Generate a KNN model
k = 5
knn = CustomKNN(n_neighbors = k)

# Train the model
knn.fit(x_train, y_train)

# 4. Predict the test data
y_pred = knn.predict(x_test)

# 5. Evaluate the model
print("Summary of the predictions made by the classifier:\n")
print(classification_report(y_test, y_pred))

# 6. Visualize the test data
plt.figure(figsize = (8, 6))
plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color = 'blue', label = 'Class 0(Test)', alpha = 0.6)
plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color = 'red', label = 'Class 1(Test)', alpha = 0.6)
plt.scatter(
    x_test[y_pred != y_test, 0], x_test[y_pred != y_test, 1], color = 'yellow', edgecolor = 'black',
    label = 'Misclassified', marker = 'x', s = 100
)
plt.title(f"KNN Classification (k={k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
