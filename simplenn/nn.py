import numpy as np


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


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


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
        self._act = softmax(x, axis=self._axis)
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
        return leaky_relu(x, self.alpha)

    def backward(self, grad):
        return leaky_relu_deriv(self._x, self.alpha) * grad

    def update_params(self, lr):
        pass


class Relu(Module):
    def __init__(self):
        self._x = None

    def forward(self, x):
        self._x = x
        return relu(x)

    def backward(self, grad):
        return relu_deriv(self._x) * grad

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
        self.bias = np.zeros((1, out_features)) if bias else None
        self._x = None
        self._dw = None
        self._db = None

    def forward(self, x):
        self._x = x
        z = x @ self.weight.T
        return z if self.bias is None else z + self.bias

    def backward(self, grad):
        self._db = np.sum(grad)
        self._dw = grad.T @ self._x
        return grad @ self.weight

    def update_params(self, lr):
        self.weight -= lr * self._dw / self._x.shape[0]
        self.bias -= lr * self._db / self._x.shape[0]


class Reshape(Module):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, input):
        self.input_shape = input.shape
        return np.reshape(input, self.output_shape)

    def backward(self, grad):
        return np.reshape(grad, self.input_shape)

    def update_params(self, lr):
        pass


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.filters = np.random.randn(
            self.kernel_size, self.kernel_size, in_channels, self.out_channels
        )
        self._filter_grads = np.zeros(self.filters.shape)
        self.input = None

    def forward(self, input):
        # `input` is 4d array with shape [batch_size, input_height, input_width, input_channels]
        self.input = input
        output_height = (input.shape[1] - self.kernel_size) // self.stride + 1
        output_width = (input.shape[2] - self.kernel_size) // self.stride + 1
        output = np.zeros(
            (input.shape[0], output_height, output_width, self.out_channels)
        )

        for batch_idx in range(input.shape[0]):
            for row in range(output_height):
                for col in range(output_width):
                    y = row * self.stride
                    x = col * self.stride

                    # patch.shape = (kernel_size, kernel_size, in_channels, out_channels)
                    patch = input[
                        batch_idx,
                        y : y + self.kernel_size,
                        x : x + self.kernel_size,
                        :,
                        np.newaxis,
                    ]
                    convolved = np.sum(patch * self.filters, axis=(0, 1, 2))
                    output[batch_idx, row, col, :] = convolved
        return output

    def backward(self, output_grad):
        self._filter_grads = np.zeros(self.filters.shape)
        for batch_idx in range(self.input.shape[0]):
            for row in range(output_grad.shape[1]):
                for col in range(output_grad.shape[2]):
                    y = row * self.stride
                    x = col * self.stride
                    # patch.shape = (kernel_size, kernel_size, in_channels)
                    patch = self.input[
                        batch_idx, y : y + self.kernel_size, x : x + self.kernel_size, :
                    ]
                    # grad.shape = (out_channels)
                    grad = output_grad[batch_idx, row, col, :]
                    # prod.shape = (kernel_size, kernel_size, in_channels, out_channels)
                    prod = (
                        patch[:, :, :, np.newaxis]
                        * grad[np.newaxis, np.newaxis, np.newaxis, :]
                    )
                    self._filter_grads += prod

        # input_grad.shape = (batch_size, input_height, input_width, in_channels)
        input_grad = np.zeros(self.input.shape)
        # padded_grad.shape = (batch_size, input_height + 2, input_width + 2, in_channels)
        padding = self.kernel_size - 1
        padded_grad = np.pad(
            output_grad,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        # print(output_grad.shape, padded_grad.shape, input_grad.shape)

        # filters_rotated.shape = (kernel_size, kernel_size, in_channels, out_channels)
        filters_rotated = np.flip(self.filters, axis=(0, 1))
        for batch_idx in range(input_grad.shape[0]):
            for row in range(input_grad.shape[1]):
                for col in range(input_grad.shape[2]):
                    y = row * self.stride
                    x = col * self.stride
                    # grad_patch.shape = (kernel_size, kernel_size, in_channels, out_channels)
                    grad_patch = padded_grad[
                        batch_idx,
                        y : y + self.kernel_size,
                        x : x + self.kernel_size,
                        np.newaxis,
                        :,
                    ]
                    # print(x, y, grad_patch.shape, filters_rotated.shape)
                    input_prod = grad_patch * filters_rotated
                    input_sum = np.sum(input_prod, axis=(0, 1, 3))
                    input_grad[batch_idx, row, col, :] += input_sum
        return input_grad

    def update_params(self, lr):
        grad_norm = np.linalg.norm(self._filter_grads)
        if grad_norm > 5:
            self._filter_grads *= 5 / grad_norm
        self.filters -= lr * self._filter_grads


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, input):
        # `input` is 4d array with shape [batch_size, input_height, input_width, in_channels]
        self.input = input
        output_height = (input.shape[1] - self.kernel_size) // self.stride + 1
        output_width = (input.shape[2] - self.kernel_size) // self.stride + 1
        output = np.zeros((input.shape[0], output_height, output_width, input.shape[3]))

        for batch_idx in range(input.shape[0]):
            for row in range(output_height):
                for col in range(output_width):
                    y = row * self.stride
                    x = col * self.stride
                    patch = input[
                        batch_idx, y : y + self.kernel_size, x : x + self.kernel_size, :
                    ]
                    pooled = np.amax(patch, axis=(0, 1))
                    output[batch_idx, row, col, :] = pooled
        return output

    def backward(self, output_grad):
        # input_grad.shape = (batch_size, input_height, input_width, in_channels)
        input_grad = np.zeros(self.input.shape)
        for batch_idx in range(self.input.shape[0]):
            for row in range(output_grad.shape[1]):
                for col in range(output_grad.shape[2]):
                    y = row * self.stride
                    x = col * self.stride

                    grad = output_grad[batch_idx, row, col, :]

                    # patch.shape = (kernel_size, kernel_size, in_channels)
                    patch = self.input[
                        batch_idx, y : y + self.kernel_size, x : x + self.kernel_size, :
                    ]
                    max_h = np.argmax(patch, axis=0)
                    max_w = np.argmax(max_h, axis=0)
                    max_indices = np.unravel_index(max_w, patch.shape[:2])
                    input_grad[
                        batch_idx,
                        y + max_indices[0],
                        x + max_indices[1],
                        np.arange(self.input.shape[3]),
                    ] += grad
        return input_grad

    def update_params(self, lr):
        pass


class MSELoss(Module):
    def __init__(self):
        self._y = None
        self._y_hat = None

    def forward(self, y_hat, y):
        self._y = y
        self._y_hat = y_hat
        return mse_loss(y_hat, y)

    def backward(self):
        return mse_loss_deriv(self._y_hat, self._y)
