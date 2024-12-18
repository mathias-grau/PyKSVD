# PyKSVD

[![PyPI version](https://badge.fury.io/py/pyksvd.svg)](https://badge.fury.io/py/pyksvd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyKSVD is a Python implementation of the K-SVD algorithm based on paper *K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation* (2006) by Michal Aharon, Michael Elad, and Alfred Bruckstein. 


## For the evaluation (Time Series 2024) :

Notebook are present in folder `notebook`
- `notebooks/ksvd_theory.ipynb`is used to understand theory behind K-SVD
- `notebooks/ksvd_synthetic.ipynb`is used to reproduce synthetic results with noising
- `notebooks/ksvd_images.ipynb`is used to reproduce image denoising and pixel update (black and white only, run example in `example_corrupted_image.py` for colored reconstructions)


## Examples

- Signal Processing : Fit a dictionay based on input signals and then transform new signals with high level sparsity. 
![Dictionary Learning](https://github.com/mathias-grau/PyKSVD/blob/main/images/paper/ksvd_paper_reconstruction.png)

- Image processing : Denoise or complete a corrupted image with missing pixels with patches. 
![Image Reconstruction with 70% missing pixels](https://github.com/mathias-grau/PyKSVD/blob/main/images/example_corrupted_image_70.png)

## Installation

```bash
pip install pyksvd
```

## Example 

For signals : (example in `example.py`)

```python
from pyksvd.pyksvd import KSVD
import numpy as np

# Define parameters
N = 1500 # number of training signals 
n = 20 # size of each signal
K = 50 # number of atoms in dict
T0 = 3 # number of non zero coefficients

Y = np.random.rand(n, N)
# Fit the model
KSVD_model = KSVD(K=K, T0=T0)
KSVD_model.fit(Y, verbose=True)
X, D = KSVD_model.X, KSVD_model.D
```

For images : (example in `example_corrupted_image.py`)

```python
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
```

## References

```bibtex
@article{Aharon2006KSVD,
  title = {K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation},
  author = {Aharon, Michal and Elad, Michael and Bruckstein, Alfred},
  year = {2006},
  journal = {IEEE Transactions on Signal Processing},
}
```