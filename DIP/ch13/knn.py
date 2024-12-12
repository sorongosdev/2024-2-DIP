import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
knn = KNeighborsClassifier(n_neighbors = k)

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
plt.title(f"KNN Classification (k={k}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
