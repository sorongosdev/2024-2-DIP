import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

x, y = make_classification(n_samples = 200, n_features = 2, n_classes = 2, n_informative = 2, n_redundant = 0,
                           random_state = 42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

plt.figure(figsize = (8, 6))
