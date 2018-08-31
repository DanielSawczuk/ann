import numpy as np
import pickle

eps = 1e-7


def MSE(real_values, approximations):
    residuum = real_values - approximations
    errors = residuum * residuum
    cost = np.sum(errors) / (2 * real_values.shape[1])
    return cost


def MSE_derivative(real_values, approximations):
   return np.subtract(approximations, real_values)


def cross_entropy(real_values, approximations):
    return -np.sum(
        real_values * np.log(approximations + eps) + (1 - real_values) * np.log(1 - approximations + eps)) / \
           real_values.shape[1]


def cross_entropy_derivative(real_values, approximations):
    return -(real_values / (approximations - eps) - (1 - real_values) / (1 - approximations + eps))


def softmax(x):
    exp = np.exp(x)
    sum = np.sum(exp, axis=0)
    return exp / sum


def log_likelihood(real_values, approximations):
    ans = np.argmax(real_values, axis=0)
    return np.sum(-np.log(approximations[ans, np.arange(approximations.shape[1])] + eps))


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def sigmoid_derivative(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)


def L1_derivative(weight, number_of_weights):
    weight = np.array([np.sign(w) for w in weight])
    return weight


def L1(weights, number_of_weights):
    abs = np.abs(weights)
    sum = [np.sum(w) for w in abs]
    sum = np.sum(sum)
    return sum / (2 * number_of_weights)


def L2_derivative(weight, number_of_weights):
    return weight


def L2(weights, number_of_weights):
    mul = weights * weights
    sum = [np.sum(w) for w in mul]
    sum = np.sum(sum)
    return sum / (2 * number_of_weights)


def vec_transpose_to_2d(vec):
    return vec[np.newaxis, :].T


def fill_all(filler, *arrays):
    for array in arrays:
        array.fill(filler)


def column_stack(data):
    x = np.column_stack(data[:, 0])
    y = np.column_stack(data[:, 1])
    return x, y


def save(network, path):
    network.callback = None
    with open(path, 'wb') as f:
        pickle.dump(network, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def placeholder(x):
    pass
