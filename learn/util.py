import numpy as np
import random
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(x, alpha=0.1):
    return np.maximum(x, x * alpha)


def leaky_relu_deriv(x, alpha=0.1):
    dx = np.ones_like(x)
    dx[alpha < 0] = alpha
    return dx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y_hat, y):
    return np.sum((y_hat - y) ** 2) / y_hat.shape[0]


def mse_loss_deriv(y_hat, y):
    return 2 * (y_hat - y) / y_hat.shape[0]


def binary_crossentropy(y, y_hat):
    epsilon = 1e-7  # to avoid division by zero errors
    y_hat = np.clip(y_hat, epsilon, 1.0 - epsilon)  # clip values to avoid NaNs
    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return np.mean(loss)
