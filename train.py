import numpy as np
from model import forward, backward, cross_entropy

def sgd_step(params, grads, lr):
    '''Update every parameter in-place'''
    for key in params:
        params[key] -= lr * grads["d" + key]

def iterate_batches(X, y, batch_size, rng):
    '''Yields (X_batch, y_batch) over one epoch, in a random order'''
    perm = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]
    
def evaluate(params, X, y):
    '''Return (mean loss, accuracy) for X, y'''
    P, cache = forward(params, X)
    loss = cross_entropy(cache["Z2"], y)
    accuracy = (P.argmax(axis=1) == y).mean() # Argmax returns index of max
    return loss, accuracy
        
def train(params, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, lr=0.1, seed=0):
    '''Train params in-place with mini-batch SGD; return a list of per-epoch metrics'''
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        for X_batch, y_batch in iterate_batches(X_train, y_train, batch_size, rng):
            _, cache = forward(params, X_batch)
            sgd_step(params, backward(params, cache, y_batch), lr)
        tr_loss, tr_acc = evaluate(params, X_train, y_train)
        va_loss, va_acc = evaluate(params, X_val, y_val)
        
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))
        print(f"epoch {epoch:2d}  train loss {tr_loss:.4f} acc {tr_acc:6.2%}")
        print(f"          val loss {va_loss:.4f} acc {va_acc:6.2%}\n")
        
    return history 