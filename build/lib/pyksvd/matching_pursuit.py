import numpy as np

def mp(D, Y, TO) :
    """
    Matching pursuit algorithm with fixed number of activations
    :param D: Dictionary matrix (n x K)
    :param Y: Signal (n)
    """
    n, K = D.shape
    x = np.zeros(K)
    r = Y
    for _ in range(TO):
        proj = D.T @ r
        k = np.argmax(np.abs(proj))
        x[k] += proj[k]
        r -= D[:, k] * proj[k]
    return x

def mp_vectorized(D, Y, TO):
    """
    Matching pursuit algorithm with fixed number of activations for multiple signals.
    :param D: Dictionary matrix (n x K)
    :param Y: Signals matrix (n x N)
    :param TO: Number of iterations (activations)
    :return: Sparse coefficients matrix (K x N)
    """
    n, K = D.shape
    N = Y.shape[1]
    
    # Initialize coefficient matrix (K x N) and residual matrix (n x N)
    X = np.zeros((K, N))
    R = Y.copy()  # residual matrix

    # Precompute D.T
    D_T = D.T

    for _ in range(TO):
        # Compute projections for all signals
        proj = D_T @ R  # (K x N)

        # Find the index of maximum projection for each signal
        k = np.argmax(np.abs(proj), axis=0)  # (N,) Index of the max projection for each signal

        # Update the coefficients at the selected indices
        for i in range(N):
            X[k[i], i] += proj[k[i], i]

        # Update residuals by subtracting the contributions of selected atoms
        for i in range(N):
            R[:, i] -= D[:, k[i]] * proj[k[i], i]

    return X


# Test the function
# K = 8
# N = 20
# n= 64

# D = np.random.randn(n, K)
# Y = np.random.randn(n, N)

# TO = 3

# X = mp_vectorized(D, Y, TO)
# print(X.shape)