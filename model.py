import numpy as np
import math

def he_normal(rng, shape):
    '''Random weights, scaled so ReLU pre-activation variance is preserved.'''
    return rng.standard_normal(shape, dtype=np.float32) * math.sqrt(2 / shape[0])

def init_params(n_in=784, n_hidden=128, n_out=10, seed=0):
    '''Return {W1, b1, W2, b2} for an n_in -> n_hidden -> n_out ReLU network.'''

    rng = np.random.default_rng(seed)
    return {
        "W1": he_normal(rng, (n_in, n_hidden)),
        "b1": np.zeros(n_hidden, dtype=np.float32),
        "W2": he_normal(rng, (n_hidden, n_out)),
        "b2": np.zeros(n_out, dtype=np.float32),
    }

def relu(Z):
    '''ReLU activation function'''
    return np.maximum(Z, 0)

def softmax(Z):
    E = np.exp(Z)
    return E / np.sum(E, axis=1, keepdims=True) # Sum along rows

print(softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])))