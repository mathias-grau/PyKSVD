import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm
from numpy.linalg import svd
from pyksvd.matching_pursuit import mp, mp_vectorized

# ksvd/ksvd.py

class KSVD:
    def __init__(self, K = 3, T0=2, max_iter=80):
        self.K = K
        self.T0 = T0
        self.MAX_ITER = max_iter
        self.n = None
        self.N = None
        self.D = None
        self.X = None
        self.E = None

        
    def fit(self, Y, verbose=False):
        """
        Fit the KSVD model to the data Y.
        """
        Y = Y.copy().astype(float)  # To avoid modifying the original data
        print("Fitting model ...")
        
        n, N = Y.shape
        self.n = n
        self.N = N
        
        # Initialize the dictionary with random columns normalized
        idxs = np.random.choice(N, self.K, replace=False)
        self.D = Y[:, idxs]
        self.D /= np.linalg.norm(self.D, axis=0)
        
        self.X = np.zeros((self.K, N))
        self.E = Y - self.D @ self.X
        
        error_t = np.linalg.norm(self.E)
        print(f"Initial error: {error_t:.4f}")
        
        pbar = tqdm(range(self.MAX_ITER), desc=f"Current error: {error_t:.4f}")
        
        for iteration in pbar:
            X_copy = self.X.copy()
            
            # Update sparse codes
            # for i in range(N):
            #     X_copy[:, i] = mp(self.D, Y[:, i], self.T0)

            X_copy = mp_vectorized(self.D, Y, self.T0)
            
            # Update error and check if the sparse codes are better
            E_t = Y - self.D @ X_copy
            if np.linalg.norm(E_t) < error_t:
                self.X = X_copy
            else:
                tqdm.write(f"Error at iteration {iteration}: {np.linalg.norm(E_t):.4f} > {error_t:.4f}; skipping update")
                break
            
            # Update dictionary
            for k in range(self.K):
                E_k = Y - self.D @ self.X + np.outer(self.D[:, k], self.X[k, :])
                x_k = self.X[k, :]
                d_k = self.D[:, k]
                indices = np.where(x_k != 0)[0]
                if len(indices) == 0:
                    continue
                
                Omega_k = np.zeros((N, len(indices)))
                for i, idx in enumerate(indices):
                    Omega_k[idx, i] = 1
                E_k_R = E_k @ Omega_k
                x_k_R = x_k[indices]
                
                U, S, V = svd(E_k_R, full_matrices=False)
                d_k_hat = U[:, 0]
                x_k_R_hat = S[0] * V[0, :]
                X_copy[k, indices] = x_k_R_hat
                self.D[:, k] = d_k_hat
            
            self.E = Y - self.D @ self.X
            error_t_plus_1 = np.linalg.norm(self.E)
            pbar.set_description(f"Current error: {error_t_plus_1:.4f}")
            
            if verbose and iteration % 5 == 0:
                tqdm.write(f"Error at iteration {iteration}: {error_t_plus_1:.4f}")
            
            if np.abs(error_t - error_t_plus_1) < 1e-5:
                tqdm.write(f"Converged at iteration {iteration}, delta: {np.abs(error_t - error_t_plus_1):.4f}")
                break
            
            error_t = error_t_plus_1
        
        print(f"Final error after fitting: {error_t_plus_1:.4f}")

    
    def fit_with_mean(self, Y, verbose=False):
        Y = Y.copy()
        print(f"Fitting model with mean ...")
        n, N = Y.shape
        self.n = n
        self.N = N
        self.D = np.zeros((self.n, self.K))
        
        # First dictionary atom is a constant for the mean
        self.D[:, 0] = 1 / np.sqrt(self.n)  # Normalize the mean atom
        self.X = np.zeros((self.K, self.N))
        
        # Compute mean of signals and assign it to the first row of X
        self.X[0, :] = np.nanmean(Y, axis=0) *  np.sqrt(self.n)
        
        # Subtract the mean from the signals
        Y_centered = Y - np.outer(self.D[:, 0], self.X[0, :])
        
        # Initialize the rest of the dictionary
        D_rest = Y_centered[:, np.random.choice(N, self.K - 1, replace=False)]
        D_rest /= np.linalg.norm(D_rest, axis=0) + 1e-8
        self.D[:, 1:] = D_rest
        
        # Initialize the rest of the sparse codes
        X_rest = np.zeros((self.K - 1, self.N))
        
        error_t = np.linalg.norm(Y_centered - self.D[:, 1:] @ X_rest)
        pbar = tqdm(range(self.MAX_ITER), desc=f"Current error: {error_t:.4f}")
        
        for _ in pbar:
            X_rest_copy = X_rest.copy()


            
            # Update sparse codes

            # for i in range(N):
            #     X_rest_copy[:, i] = mp(self.D[:, 1:], Y_centered[:, i], self.T0 - 1)
            X_rest_copy = mp_vectorized(self.D[:, 1:], Y_centered, self.T0 - 1)
            
            # Update the dictionary atoms
            for k in range(self.K - 1):
                E_k = Y_centered - self.D[:, 1:] @ X_rest + np.outer(self.D[:, k + 1], X_rest[k, :])
                indices = np.where(X_rest[k, :] != 0)[0]
                if len(indices) == 0:
                    continue
                
                Omega_k = np.zeros((N, len(indices)))
                for i, idx in enumerate(indices):
                    Omega_k[idx, i] = 1
                E_k_R = E_k @ Omega_k
                U, S, V = svd(E_k_R, full_matrices=False)
                d_k_hat = U[:, 0]
                x_k_R_hat = S[0] * V[0, :]
                X_rest[k, indices] = x_k_R_hat
                self.D[:, k + 1] = d_k_hat
            
            # Recalculate error
            self.E = Y_centered - self.D[:, 1:] @ X_rest
            error_t_plus_1 = np.linalg.norm(self.E)
            pbar.set_description(f"Current error: {error_t_plus_1:.4f}")
            
            if np.abs(error_t - error_t_plus_1) < 1e-5:
                tqdm.write(f"Converged at iteration {_}, delta: {np.abs(error_t - error_t_plus_1):.4f}")
                break
            
            error_t = error_t_plus_1
        
        # Update the final dictionary and sparse codes
        self.D[:, 1:] = D_rest
        self.X[1:, :] = X_rest
        self.E = Y - self.D @ self.X
        print(f"Final error after fitting: {np.linalg.norm(self.E):.4f}")

                
    def transform(self, Y):
        """
        Transform data Y into the sparse code representation using the learned dictionary.
        """
        Y = Y.copy()  # Avoid modifying the original data
        n_t, N_t = Y.shape
        assert n_t == self.n, f"Expected Y of shape {(self.n, N_t)}, but got {(n_t, N_t)}"
        
        print(f"Transforming data ...")
        
        X_t = np.zeros((self.K, N_t))
        
        # for i in tqdm(range(N_t)):
        #     X_t[:, i] = mp(self.D, Y[:, i], self.T0)
        
        X_t = mp_vectorized(self.D, Y, self.T0)

        # Calculate reconstruction error
        E_t = Y - self.D @ X_t
        error_t_plus_1 = np.linalg.norm(E_t)
        print(f"Error after transformation: {error_t_plus_1:.4f}")
        
        return X_t, self.D


    def transform_with_mean(self, Y):
        Y = Y.copy()
        print(f"Transforming data with mean ...")
        n_t, N_t = Y.shape
        assert n_t == self.n, f"Expected Y of shape {(self.n, N_t)}, but got {(n_t, N_t)}"
        
        # Allocate the sparse code matrix
        X_t = np.zeros((self.K, N_t))
        
        # First coefficient row corresponds to the mean
        X_t[0, :] = np.mean(Y, axis=0) * np.sqrt(self.n)
        
        # Subtract the mean from the signals
        Y_centered = Y - np.outer(self.D[:, 0], X_t[0, :])
        
        # Transform using the rest of the dictionary
        # for i in tqdm(range(N_t)):
        #     X_t[1:, i] = mp(self.D[:, 1:], Y_centered[:, i], self.T0 - 1)
        X_t[1:, :] = mp_vectorized(self.D[:, 1:], Y_centered, self.T0 - 1)
        
        # Calculate the reconstruction error
        E_t = Y - self.D @ X_t
        error_t_plus_1 = np.linalg.norm(E_t)
        print(f"Final error after transformation: {error_t_plus_1:.4f}")
        return X_t, self.D

    
    def fit_transform(self, Y):
        self.fit(Y)
        return self.transform(Y), self.D

    def fit_transform_with_mean(self, Y):
        self.fit_with_mean(Y)
        return self.transform_with_mean(Y), self.D

    def transform_with_mean_signal_with_null_values(self, Y):
        Y = Y.copy()  # Avoid modifying original data
        n_t, N_t = Y.shape

        # Initialize sparse code matrix
        X_t = np.zeros((self.K, N_t))

        for i in range(N_t):
            Y_i = Y[:, i]  # i-th signal

            # Identify non-null indices (either NaN or 0)
            non_null_idx = np.where(~np.isnan(Y_i) & (Y_i != 0))[0]
            Y_i_non_null = Y_i[non_null_idx]  # Non-null values

            if len(non_null_idx) == 0:
                print(f"Warning: Signal {i} has no non-null values. Skipping.")
                continue  # Skip this signal if no non-null values are found

            # Compute mean of non-null values
            mean_non_null_i = np.nanmean(Y_i_non_null)  # Mean of non-null values

            # Store the mean in the first coefficient
            X_t[0, i] = mean_non_null_i * np.sqrt(self.n)

            # Center the signal by subtracting the mean of non-null values
            Y_i_centered_non_null = Y_i_non_null - mean_non_null_i

            # Construct the dictionary corresponding to the non-null values
            D_i = self.D[non_null_idx, 1:].copy()  # Using the dictionary rows corresponding to non-null values

            # Perform matching pursuit on the centered signal using the non-null dictionary subset
            X_t_i = mp(D_i, Y_i_centered_non_null, self.T0 - 1)

            # Store the resulting coefficients for the rest of the coefficients
            X_t[1:, i] = X_t_i

        # Calculate the reconstruction error
        E = Y - self.D @ X_t
        error = np.linalg.norm(E)
        print(f"Error after transformation: {error:.4f}")

        return X_t, self.D
