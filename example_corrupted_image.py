import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pyksvd.functions import train_ksvd_models, corrupt_image, reconstruct_image

# Define parameters
patch_size = 8
image_size = 256
K = 441  # number of atoms in dict
T0 = 10  # number of non zero coefficients
REMOVE_PIXELS_RATIO = 0.7

# Directories
train_dir = 'data/train/impressionism/'
test_image_path = 'data/test/impressionism/134.jpg'

# Train KSVD models
ksvd_models = train_ksvd_models(train_dir, patch_size, K, T0)

# Load test image
test_image = Image.open(test_image_path)
test_image = test_image.resize((image_size, image_size))
test_image_array = np.array(test_image, dtype=np.float32) / 255.0

# Corrupt image
corrupted_image, mask = corrupt_image(test_image_array, REMOVE_PIXELS_RATIO)

# Reconstruct image
reconstructed_image = reconstruct_image(corrupted_image, ksvd_models, patch_size)

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(test_image_array)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(corrupted_image)
axs[1].set_title(f"Corrupted Image ({REMOVE_PIXELS_RATIO * 100:.0f}% pixels removed)")
axs[1].axis("off")

axs[2].imshow(reconstructed_image)
axs[2].set_title("Reconstructed Image")
axs[2].axis("off")

plt.tight_layout()
plt.savefig(f"images/example_corrupted_image_{REMOVE_PIXELS_RATIO*100:.0f}.png")
plt.show()
