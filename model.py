import numpy as np
import math

def he_normal(rng, shape):
    '''Random weights, scaled so ReLU pre-activation variance is preserved'''
    return rng.standard_normal(shape, dtype=np.float32) * math.sqrt(2 / shape[0])

def init_params(n_in=784, n_hidden=128, n_out=10, seed=0):
    '''Return {W1, b1, W2, b2} for an n_in -> n_hidden -> n_out ReLU network'''

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
    '''Row-wise softmax, shifted by row max to avoid overflow'''
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / np.sum(E, axis=1, keepdims=True) # Sum along rows

def log_softmax(Z): 
    '''Row-wise log softmax, computed without forming negligible probabilities
    Calculates log(softmax(Z)), passing log through softmax using log laws'''
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))

def forward(params, X):
    '''Feed X through the network; return (P, cache) with cache holding X, Z1, A1.'''
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    P = softmax(Z2)
    return P, {"X" : X, "Z1" : Z1, "A1" : A1, "Z2": Z2}

def cross_entropy(logits, y):
    '''Mean cross-entropy of integer labels against network logits (pre-softmax output)'''    
    return -log_softmax(logits)[np.arange(len(y)), y].mean()

def backward(params, cache, y):
    '''Return {dW1, db1, dW2, db2}, the mean of the per-example gradients over the batch'''
    W2 = params["W2"]
    X, Z1, A1, Z2 = cache["X"], cache["Z1"], cache["A1"], cache["Z2"]
    N = len(y)
    
    # dL/dZ2 = (P - y) / N
    dZ2 = softmax(Z2)
    dZ2[np.arange(len(y)), y] -= 1
    dZ2 /= N 
    
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ W2.T
    
    dZ1 = dA1 * (Z1 > 0)
    
    dW1 = np.transpose(X) @ dZ1
    db1 = dZ1.sum(axis=0)

    return {"dW1" : dW1, "db1" : db1, "dW2" : dW2, "db2" : db2}