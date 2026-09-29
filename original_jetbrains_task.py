"""
original_jetbrains_task.py
==========================
Original task solution script completed as part of the JetBrains Academy
'Mastering Python Libraries – NumPy' Course.

This script demonstrates:
1. Loading an RGB image with Matplotlib (`plt.imread`).
2. Rescaling pixel values from [0, 255] to [0.0, 1.0].
3. Permuting array axes from (H, W, 3) to (3, H, W) via `np.transpose`
   to enable multi-dimensional batched SVD.
4. Computing Singular Value Decomposition with `numpy.linalg.svd`.
5. Reconstructing the diagonal Sigma matrix using `np.fill_diagonal`.
6. Performing low-rank approximation (k=10) with matrix multiplication `@`.
7. Clipping floating-point drift to [0.0, 1.0] and transposing back to (H, W, 3) for display.
"""

from numpy import linalg
import numpy as np
import matplotlib.pyplot as plt
import os

# Load image
img_path = os.path.join(os.path.dirname(__file__), "assets", "horse.jpg")
img = plt.imread(img_path)

# Step 1: Rescale data to [0, 1]
img_rescaled = img / 255.0

# Step 2: Permute axes to shape (Channels, Height, Width) -> (3, 408, 612)
img_array_transposed = np.transpose(img_rescaled, (2, 0, 1))

# Step 3: Compute Batched SVD across all 3 color channels simultaneously
U, s, Vt = linalg.svd(img_array_transposed)

# Step 4: Rebuild diagonal matrix Sigma of dimensions (3, 408, 612)
Sigma = np.zeros((3, 408, 612))
for j in range(3):
    np.fill_diagonal(Sigma[j, :, :], s[j, :])

# Step 5: Full reconstruction test (U @ Sigma @ Vt)
reconstructed = U @ Sigma @ Vt
reconstructed = np.clip(reconstructed, 0.0, 1.0)

# Step 6: Low-Rank Approximation using top k = 10 singular values
k = 10
approx_img = U[:, :, :k] @ Sigma[:, :k, :k] @ Vt[:, :k, :]

# Permute axes back to (Height, Width, Channels) for display
display_img = np.transpose(approx_img, (1, 2, 0))

if __name__ == '__main__':
    print("Approximation shape:", approx_img.shape)
    print("Display image shape:", display_img.shape)
    plt.figure(figsize=(8, 6))
    plt.title(f"JetBrains Academy SVD Task - Rank k = {k} Approximation")
    plt.imshow(display_img)
    plt.axis("off")
    plt.show()
