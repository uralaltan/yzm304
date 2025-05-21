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


def initialize_parameters_2layer(n_x, n_h, n_y=1):
    np.random.seed(42)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def forward_propagation_2layer(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X.T) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[0]
    logprobs = - (np.dot(np.log(A2), Y) + np.dot(np.log(1 - A2), (1 - Y)))
    cost = float(np.squeeze(logprobs) / m)
    return cost


def backpropagation_2layer(X, Y, cache, parameters):
    m = X.shape[0]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Y = Y.T

    dZ2 = A2 - Y
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * tanh_derivative(A1)
    dW1 = (1.0 / m) * np.dot(dZ1, X)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


def update_parameters_2layer(parameters, grads, learning_rate=0.01):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters


def nn_model_2layer(X, Y, n_x, n_h, n_y=1, n_steps=1000, learning_rate=0.01, print_cost=True):
    parameters = initialize_parameters_2layer(n_x, n_h, n_y)

    for i in range(n_steps):
        A2, cache = forward_propagation_2layer(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backpropagation_2layer(X, Y, cache, parameters)
        parameters = update_parameters_2layer(parameters, grads, learning_rate=learning_rate)

        if print_cost and i % 1000 == 0:
            print(f"Step {i}, Cost: {cost:.6f}")

    return parameters


def predict_2layer(parameters, X):
    A2, _ = forward_propagation_2layer(X, parameters)
    return (A2 > 0.5).astype(int).flatten()


parameters_2layer = nn_model_2layer(
    X_train, y_train,
    n_x=X_train.shape[1],
    n_h=6,
    n_y=1,
    n_steps=800,
    learning_rate=0.01
)

y_pred_2layer = predict_2layer(parameters_2layer, X_test)
y_true_2layer = y_test.flatten()

acc = accuracy_score(y_true_2layer, y_pred_2layer)
prec = precision_score(y_true_2layer, y_pred_2layer)
rec = recall_score(y_true_2layer, y_pred_2layer)
f1 = f1_score(y_true_2layer, y_pred_2layer)
cm = confusion_matrix(y_true_2layer, y_pred_2layer)

print("\n=== 2-Layer Model (tanh) Results ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true_2layer, y_pred_2layer))
