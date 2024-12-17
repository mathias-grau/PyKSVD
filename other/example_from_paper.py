from ksvd.ksvd import KSVD
import numpy as np
import matplotlib.pyplot as plt
# Specify the color palette and linestyles
color_palette = plt.get_cmap("plasma")  # Diverging colormap
linestyle_true = '-'  # Solid line for true signal
linestyle_reconstructed = '--'  # Dashed line for reconstructed signal
SNR_dB = 20


# Define parameters
N = 1500 # number of training signals 
n = 20 # size of each signal
K = 50 # number of atoms in dict
T0 = 3 # number of non zero coefficients


D_init = np.zeros((n, K))
frequencies = np.sort(np.random.uniform(0.3, 3, K))  # Random frequencies between 0.5 and 3 Hz
time = np.linspace(0, 1, n)
for i in range(K):
    signal = np.sin(2 * np.pi * frequencies[i] * time) 
    D_init[:, i] = signal

# Create N signals y which are linear combinations of 3 random signals with uniformly distributed coefficients and then normalised 
Y = np.zeros((n, N))
for i in range(N):
    idxs = np.random.choice(K, 3, replace=False)
    coeffs = np.random.uniform(0, 1, 3)
    coeffs /= np.linalg.norm(coeffs)
    signal = D_init[:, idxs] @ coeffs
    signal /= np.linalg.norm(signal)
    noise = np.random.normal(0, 1, n)
    noise /= np.linalg.norm(noise)
    noise *= np.linalg.norm(signal) / 10**(SNR_dB/20)
    Y[:, i] = signal


# Fit the model
KSVD_model = KSVD(K=K, T0=T0)
KSVD_model.fit(Y, verbose=True)
X, D = KSVD_model.X, KSVD_model.D

# plot the first true signals and the reconstructed signals
N_SIGNALS = min(4, N)
plt.figure(figsize=(10, 5))
colors = [color_palette(i / N_SIGNALS) for i in range(N_SIGNALS)]
for i in range(N_SIGNALS):
    plt.plot(Y[:, i], label=f'True signal {i}', color=colors[i], linestyle=linestyle_true, alpha=0.5)
    plt.plot(D @ X[:, i], label=f'Reconstructed signal {i}', color=colors[i], linestyle=linestyle_reconstructed, alpha=0.5)
plt.title(f'Signals with reconstruction using KSVD')
plt.legend()
plt.grid()
plt.savefig("./images/paper/ksvd_paper_reconstruction.png")
plt.show()

# plot initial dictionary and learned dictionary
plt.figure(figsize=(10, 5))
plt.plot(D_init, label="Initial dictionary", linestyle=linestyle_true)
plt.plot(D, label="Learned dictionary", linestyle=linestyle_reconstructed)
plt.title("Initial and learned dictionary")
plt.legend()
plt.grid()
plt.savefig("./images/paper/ksvd_paper_dict.png")
plt.show()