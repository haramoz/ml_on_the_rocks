import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
np.set_printoptions(precision=2)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def plt_layer_relu(X, y, W, b, classes):
    """
    Plots the decision boundary of the first ReLU-activated layer in a neural network.
    
    Parameters:
    - X: Training data (features)
    - y: Training labels
    - W: Weights of the first layer
    - b: Biases of the first layer
    - classes: List of class names
    """
    # Generate a mesh grid covering the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Flatten the mesh grid to pass through the network
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute the activation output of the first layer
    Z = np.dot(grid_points, W) + b
    A = np.maximum(Z, 0)  # ReLU activation

    # Assign classes based on the maximum activation value
    class_predictions = np.argmax(A, axis=1)
    class_predictions = class_predictions.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, class_predictions, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Scatter plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # Add legend for classes
    handles, labels = scatter.legend_elements()
    plt.legend(handles, classes, title="Classes")

    # Plot settings
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("First Layer Decision Boundary with ReLU Activation")
    plt.show()


# To generate data for training
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)

### Needs to be replaced with plt implementation
plt_mc(X_train,y_train,classes, centers, std=std)

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        Dense(2, activation = 'relu',   name = "L1"),
        Dense(4, activation = 'linear', name = "L2")
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

model.fit(
    X_train,y_train,
    epochs=200
)

### Needs to be replaced with plt implementation
plt_cat_mc(X_train, y_train, model, classes)

# gather the trained parameters from the first layer
l1 = model.get_layer("L1")
W1,b1 = l1.get_weights()

# plot the function of the first layer
plt_layer_relu(X_train, y_train.reshape(-1,), W1, b1, classes)

