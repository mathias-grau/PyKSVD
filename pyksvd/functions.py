import os
import glob
import numpy as np
from PIL import Image
from pyksvd.pyksvd import KSVD

def extract_patches_custom(image, patch_size):
    """
    Extract patches from an image with precise handling
    
    :param image: Input image (numpy array)
    :param patch_size: Size of patches to extract
    :return: Array of patches
    """
    h, w, c = image.shape
    patches = []
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # Ensure we get exactly patch_size x patch_size patches
            if i + patch_size <= h and j + patch_size <= w:
                patch = image[i:i+patch_size, j:j+patch_size, :]
                if patch.shape == (patch_size, patch_size, 3):
                    patches.append(patch)
    
    return np.array(patches)

def train_ksvd_models(image_dir, patch_size, K, T0, image_size = 256):
    """
    Train KSVD models for each color channel
    
    :param image_dir: Directory with training images
    :param patch_size: Size of patches
    :param K: Dictionary size
    :param T0: Sparsity
    :return: List of trained KSVD models
    """
    # Load images
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    
    # Store patches for each channel
    channel_patches = [[] for _ in range(3)]
    
    for image_file in image_files:
        # Read and preprocess image
        img = Image.open(image_file)
        img = img.resize((image_size, image_size))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Extract patches
        patches = extract_patches_custom(img_array, patch_size)
        
        # Separate channels
        for channel in range(3):
            channel_patches[channel].append(
                patches[..., channel].reshape(-1, patch_size * patch_size)
            )
    
    # Concatenate patches for each channel
    ksvd_models = []
    for channel in range(3):
        # Combine patches from all images
        channel_data = np.concatenate(channel_patches[channel])
        
        # Train KSVD model
        model = KSVD(K=K, T0=T0)
        model.fit_with_mean(channel_data.T)
        ksvd_models.append(model)
    
    return ksvd_models

def corrupt_image(image, remove_ratio):
    """
    Corrupt image by removing pixels
    
    :param image: Input image
    :param remove_ratio: Ratio of pixels to remove
    :return: Corrupted image
    """
    corrupted_image = image.copy()
    mask = np.random.random(image.shape[:2]) < remove_ratio
    corrupted_image[mask] = 0
    return corrupted_image, mask

def reconstruct_image(corrupted_image, ksvd_models, patch_size):
    """
    Reconstruct image using learned dictionaries
    
    :param corrupted_image: Image with missing pixels
    :param ksvd_models: Trained KSVD models for each channel
    :param patch_size: Size of patches
    :return: Reconstructed image
    """
    # Extract corrupted patches
    corrupted_patches = extract_patches_custom(corrupted_image, patch_size)
    
    # Prepare reconstructed image
    reconstructed_image = np.zeros_like(corrupted_image)
    
    # Reconstruct each channel
    for channel_idx, ksvd_model in enumerate(ksvd_models):
        # Extract and flatten channel patches
        channel_patches = corrupted_patches[..., channel_idx].reshape(-1, patch_size * patch_size)
        
        # Sparse coding
        X_corrupted, _ = ksvd_model.transform_with_mean_signal_with_null_values(channel_patches.T)
        
        # Reconstruct patches
        Y_reconstructed = ksvd_model.D @ X_corrupted
        reconstructed_patches = Y_reconstructed.T.reshape(-1, patch_size, patch_size)
        
        # Reconstruct channel
        h, w = corrupted_image.shape[:2]
        channel_reconstructed = np.zeros((h, w), dtype=float)
        
        patch_idx = 0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                if i + patch_size <= h and j + patch_size <= w and patch_idx < len(reconstructed_patches):
                    channel_reconstructed[i:i+patch_size, j:j+patch_size] = reconstructed_patches[patch_idx]
                    patch_idx += 1
        
        # Assign reconstructed channel
        reconstructed_image[..., channel_idx] = channel_reconstructed
    
    return np.clip(reconstructed_image, 0, 1)