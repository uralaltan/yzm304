import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

df = pd.read_csv("BankNote_Authentication.csv")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X, y = df.iloc[:, :-1], df.iloc[:, -1]

X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("X train shape:", X_train.shape)
print("X test shape:", X_test.shape)
print("y train shape:", y_train.shape)
print("y test shape:", y_test.shape)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def tanh(Z):
    return np.tanh(Z)


def tanh_derivative(A):
    return 1 - np.power(A, 2)


def initialize_parameters_3layer(n_x, n_h1, n_h2, n_y=1):
    np.random.seed(42)

    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros((n_h1, 1))

    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros((n_h2, 1))

    W3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    return parameters


def forward_propagation_3layer(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X.T) + b1
    A1 = tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = tanh(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3
    }

    return A3, cache


def compute_cost(A3, Y):
    m = Y.shape[0]

    logprobs = - (np.dot(np.log(A3), Y) + np.dot(np.log(1 - A3), (1 - Y)))
    cost = np.squeeze(logprobs) / m
    cost = float(cost)
    return cost


def backpropagation_3layer(X, Y, cache, parameters):
    m = X.shape[0]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    Y = Y.T
    dZ3 = A3 - Y

    dW3 = (1.0 / m) * np.dot(dZ3, A2.T)
    db3 = (1.0 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = np.dot(W3.T, dZ3) * tanh_derivative(A2)

    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * tanh_derivative(A1)

    dW1 = (1.0 / m) * np.dot(dZ1, X)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3
    }
    return grads


def update_parameters_3layer(parameters, grads, learning_rate=0.01):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    parameters = {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "W3": W3, "b3": b3
    }
    return parameters


def nn_model_3layer(X, Y, n_x, n_h1, n_h2, n_y=1, n_steps=1000, learning_rate=0.01, print_cost=True):
    parameters = initialize_parameters_3layer(n_x, n_h1, n_h2, n_y)

    for i in range(n_steps):
        A3, cache = forward_propagation_3layer(X, parameters)

        cost = compute_cost(A3, Y)

        grads = backpropagation_3layer(X, Y, cache, parameters)

        parameters = update_parameters_3layer(parameters, grads, learning_rate=learning_rate)

        if print_cost and i % 1000 == 0:
            print(f"Step {i}, Cost: {cost:.6f}")

    return parameters


def predict_3layer(parameters, X):
    A3, _ = forward_propagation_3layer(X, parameters)
    return (A3 > 0.5).astype(int).flatten()


parameters_3layer = nn_model_3layer(
    X_train, y_train,
    n_x=X_train.shape[1],
    n_h1=6,
    n_h2=6,
    n_y=1,
    n_steps=4200,
    learning_rate=0.01
)

y_pred_3layer = predict_3layer(parameters_3layer, X_test)
y_true_3layer = y_test.flatten()

acc = accuracy_score(y_true_3layer, y_pred_3layer)
prec = precision_score(y_true_3layer, y_pred_3layer)
rec = recall_score(y_true_3layer, y_pred_3layer)
f1 = f1_score(y_true_3layer, y_pred_3layer)
cm = confusion_matrix(y_true_3layer, y_pred_3layer)

print("\n=== 3-Layer Model (tanh) Results ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true_3layer, y_pred_3layer))
