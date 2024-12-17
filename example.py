from pyksvd.pyksvd import KSVD
import numpy as np
import matplotlib.pyplot as plt
# Specify the color palette and linestyles
color_palette = plt.get_cmap("plasma")  # Diverging colormap
linestyle_true = '-'  # Solid line for true signal
linestyle_reconstructed = '--'  # Dashed line for reconstructed signal

# Define parameters
N = 50 # number of training signals 
n = 20 # size of each signal
K = 10 # number of atoms in dict
T0 = 3 # number of non zero coefficients

Y = np.zeros((n, N))
frequencies = np.sort(np.random.uniform(2, 7, N))  # Random frequencies between 0.5 and 3 Hz
time = np.linspace(0, 1, n)
for i in range(N):
    signal = np.sin(2 * np.pi * frequencies[i] * time) 
    Y[:, i] = signal

# Fit the model
KSVD_model = KSVD(K=K, T0=T0)
KSVD_model.fit(Y, verbose=True)
X, D = KSVD_model.X, KSVD_model.D

# plot the first true signals and the reconstructed signals
N_SIGNALS = min(6, N)
plt.figure(figsize=(10, 5))
colors = [color_palette(i / N_SIGNALS) for i in range(N_SIGNALS)]
for i in range(N_SIGNALS):
    plt.plot(Y[:, i], label=f'True signal {i}', color=colors[i], linestyle=linestyle_true, alpha=0.5)
    plt.plot(D @ X[:, i], label=f'Reconstructed signal {i}', color=colors[i], linestyle=linestyle_reconstructed, alpha=0.5)
plt.title(f'Signals with reconstruction using KSVD')
plt.legend()
plt.grid()
plt.savefig("./images/ksvd_reconstruction.png")
plt.show()