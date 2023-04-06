import numpy as np
import util


class Module:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update_params(self, lr):
        raise NotImplementedError


class Softmax(Module):
    def __init__(self, axis=-1):
        self._act = None
        self._axis = axis

    def forward(self, x):
        self._act = util.softmax(x, axis=self._axis)
        return self._act

    def backward(self, grad):
        # TODO(d): figure out how to deal with this when there's a cross-entropy loss function
        return grad

    def update_params(self, lr):
        pass


class LeakyRelu(Module):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self._x = None

    def forward(self, x):
        self._x = x
        return util.leaky_relu(x, self.alpha)

    def backward(self, grad):
        return util.leaky_relu_deriv(self._x, self.alpha) * grad

    def update_params(self, lr):
        pass


class Sequential(Module):
    def __init__(self, layers):
        super().__init__(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_params(self, lr):
        for layer in self.layers:
            layer.update_params(lr)


class LinearLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.zeros((out_features, 1)) if bias else None
        self._x = None
        self._dw = None
        self._db = None

    def forward(self, x):
        self._x = x
        z = self.weight @ x
        return z if self.bias is None else z + self.bias

    def backward(self, grad):
        self._db = np.sum(grad)
        self._dw = grad @ self._x.T
        return self.weight.T @ grad

    def update_params(self, lr):
        self.weight -= lr * self._dw
        self.bias -= lr * self._db


class MSELoss(Module):
    def __init__(self):
        self._y = None
        self._y_hat = None

    def forward(self, y_hat, y):
        self._y = y
        self._y_hat = y_hat
        return util.mse_loss(y_hat, y)

    def backward(self):
        return util.mse_loss_deriv(self._y_hat, self._y)
