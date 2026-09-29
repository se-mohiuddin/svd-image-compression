"""
svd_compressor.py
=================
Vectorized Image Compression and Low-Rank Matrix Approximation
using Singular Value Decomposition (SVD) and NumPy.
"""

from typing import List, Tuple, Dict, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class SVDImageCompressor:
    """
    Performs low-rank matrix approximation on 2D (grayscale) and 3D (RGB) images
    using NumPy's vectorized linear algebra routines.
    """

    def __init__(self, image_path: str):
        """
        Loads and initializes an image for SVD processing.

        Args:
            image_path: Path to the target image file.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        self.image_path = image_path
        # Read image and normalize pixel values to [0.0, 1.0]
        raw_img = plt.imread(image_path)
        if raw_img.dtype == np.uint8:
            self.img_normalized = raw_img.astype(np.float64) / 255.0
        else:
            self.img_normalized = raw_img.astype(np.float64)

        # Handle RGBA by discarding alpha channel if present
        if self.img_normalized.ndim == 3 and self.img_normalized.shape[2] == 4:
            self.img_normalized = self.img_normalized[:, :, :3]

        self.is_color = self.img_normalized.ndim == 3
        self.height, self.width = self.img_normalized.shape[:2]
        self.num_channels = self.img_normalized.shape[2] if self.is_color else 1
        self.max_rank = min(self.height, self.width)

        # Cache for decomposition matrices
        self.U: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None

        self._decompose()

    def _decompose(self) -> None:
        """
        Computes the Singular Value Decomposition (SVD).
        For color images, axes are permuted from (H, W, C) to (C, H, W) to leverage
        NumPy's vectorized batch linear algebra without any for-loops.
        """
        if self.is_color:
            # Reorder axes: (Height, Width, Channels) -> (Channels, Height, Width)
            # Shape becomes: (3, H, W)
            tensor_stack = np.transpose(self.img_normalized, (2, 0, 1))
            # NumPy batched SVD computes decomposition over all 3 channels in parallel
            self.U, self.s, self.Vt = np.linalg.svd(tensor_stack, full_matrices=False)
        else:
            # 2D Grayscale matrix
            self.U, self.s, self.Vt = np.linalg.svd(self.img_normalized, full_matrices=False)

    def reconstruct(self, k: int, use_fill_diagonal: bool = False) -> np.ndarray:
        """
        Reconstructs the image using the top-k singular values (Rank-k approximation).

        Args:
            k: Number of singular values / rank components to retain (1 <= k <= max_rank).
            use_fill_diagonal: If True, uses the explicit Sigma matrix construction
                               with np.fill_diagonal (pedagogical method).
                               If False, uses efficient vectorized scaling.

        Returns:
            Reconstructed image array with values clipped to [0.0, 1.0].
        """
        k = max(1, min(k, self.max_rank))

        if self.is_color:
            if use_fill_diagonal:
                # Pedagogical method (as taught in JetBrains Academy course)
                Sigma = np.zeros((self.num_channels, self.max_rank, self.max_rank))
                for c in range(self.num_channels):
                    np.fill_diagonal(Sigma[c, :, :], self.s[c, :])
                # Truncate and multiply: (3, H, k) @ (3, k, k) @ (3, k, W) -> (3, H, W)
                approx_tensor = self.U[:, :, :k] @ Sigma[:, :k, :k] @ self.Vt[:, :k, :]
            else:
                # Optimized vectorized broadcast scaling: (U_k * s_k) @ Vt_k
                # Avoids allocating a large (3, M, N) zeros matrix
                scaled_U = self.U[:, :, :k] * self.s[:, np.newaxis, :k]
                approx_tensor = scaled_U @ self.Vt[:, :k, :]

            # Permute back to standard display format: (Channels, Height, Width) -> (Height, Width, Channels)
            reconstructed = np.transpose(approx_tensor, (1, 2, 0))
        else:
            # Grayscale 2D reconstruction
            if use_fill_diagonal:
                Sigma = np.diag(self.s[:k])
                reconstructed = self.U[:, :k] @ Sigma @ self.Vt[:k, :]
            else:
                scaled_U = self.U[:, :k] * self.s[:k]
                reconstructed = scaled_U @ self.Vt[:k, :]

        # Clip values to eliminate numerical floating-point drift outside [0, 1]
        reconstructed = np.clip(reconstructed, 0.0, 1.0)
        return reconstructed

    def compute_metrics(self, k: int) -> Dict[str, float]:
        """
        Calculates mathematical metrics for a given rank-k approximation:
        - Parameter count and compression ratio
        - Frobenius norm relative error: ||A - A_k||_F / ||A||_F
        - Cumulative spectral energy / variance preserved: sum(s_k^2) / sum(s^2)

        Args:
            k: Number of singular values retained.

        Returns:
            Dictionary with computed metrics.
        """
        k = max(1, min(k, self.max_rank))

        # Storage footprint:
        # Original: H * W * C
        # SVD rank-k: k * (H + W + 1) * C
        original_params = self.height * self.width * self.num_channels
        svd_params = k * (self.height + self.width + 1) * self.num_channels
        compression_ratio = original_params / svd_params
        space_savings_pct = (1.0 - (svd_params / original_params)) * 100.0

        # Cumulative energy / variance explained:
        # Based on sum of squared singular values (Frobenius energy)
        if self.is_color:
            total_energy = np.sum(self.s ** 2)
            retained_energy = np.sum(self.s[:, :k] ** 2)
        else:
            total_energy = np.sum(self.s ** 2)
            retained_energy = np.sum(self.s[:k] ** 2)
        energy_pct = (retained_energy / total_energy) * 100.0

        # Reconstruction error (Frobenius norm)
        reconstructed = self.reconstruct(k)
        diff = self.img_normalized - reconstructed
        frobenius_error = float(np.linalg.norm(diff) / np.linalg.norm(self.img_normalized))

        return {
            "k": k,
            "original_elements": original_params,
            "svd_elements": svd_params,
            "compression_ratio": float(compression_ratio),
            "space_savings_pct": float(space_savings_pct),
            "energy_preserved_pct": float(energy_pct),
            "relative_frobenius_error": frobenius_error,
        }

    def plot_singular_values(self, save_path: Optional[str] = None) -> None:
        """
        Plots the singular value spectrum (decay curve) and cumulative energy.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        if self.is_color:
            colors = ["#e74c3c", "#2ecc71", "#3498db"]
            labels = ["Red Channel", "Green Channel", "Blue Channel"]
            for c in range(3):
                ax1.semilogy(self.s[c], color=colors[c], label=labels[c], lw=1.8)
                cumulative_var = np.cumsum(self.s[c] ** 2) / np.sum(self.s[c] ** 2) * 100.0
                ax2.plot(cumulative_var, color=colors[c], label=labels[c], lw=1.8)
        else:
            ax1.semilogy(self.s, color="#2c3e50", label="Singular Values", lw=2.0)
            cumulative_var = np.cumsum(self.s ** 2) / np.sum(self.s ** 2) * 100.0
            ax2.plot(cumulative_var, color="#2c3e50", label="Singular Values", lw=2.0)

        # Plot 1: Spectrum
        ax1.set_title("Singular Value Decay (Log Scale)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Singular Value Index (Rank)", fontsize=10)
        ax1.set_ylabel(r"Singular Value $\sigma_i$ (Log Scale)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        # Plot 2: Cumulative energy
        ax2.set_title("Cumulative Spectral Energy Preserved (%)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Singular Value Index (Rank)", fontsize=10)
        ax2.set_ylabel("Preserved Energy (%)", fontsize=10)
        ax2.axhline(90, color="orange", linestyle=":", label="90% Threshold")
        ax2.axhline(99, color="red", linestyle=":", label="99% Threshold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[+] Singular value spectrum saved to: {save_path}")
        plt.close(fig)

    def plot_comparison_grid(
        self,
        k_values: List[int],
        save_path: Optional[str] = None
    ) -> None:
        """
        Generates a comparative grid displaying the original image alongside
        reconstructions at multiple rank-k levels with quality metrics.
        """
        num_images = len(k_values) + 1
        cols = 3
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows))
        axes = np.array(axes).reshape(-1)

        # Show Original Image
        cmap = None if self.is_color else "gray"
        axes[0].imshow(self.img_normalized, cmap=cmap)
        axes[0].set_title(
            f"Original Image\nShape: {self.img_normalized.shape}\nRank: {self.max_rank} (100% data)",
            fontsize=10,
            fontweight="bold"
        )
        axes[0].axis("off")

        # Show Rank-k Approximations
        for idx, k in enumerate(k_values, start=1):
            approx = self.reconstruct(k)
            metrics = self.compute_metrics(k)
            savings = metrics["space_savings_pct"]
            energy = metrics["energy_preserved_pct"]

            axes[idx].imshow(approx, cmap=cmap)
            title = (
                f"Rank k = {k}\n"
                f"Energy: {energy:.1f}% | Savings: {savings:.1f}%\n"
                f"Frobenius Error: {metrics['relative_frobenius_error']:.3f}"
            )
            axes[idx].set_title(title, fontsize=9)
            axes[idx].axis("off")

        # Hide any unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis("off")

        plt.suptitle(
            "SVD Low-Rank Image Compression & Reconstruction Analysis",
            fontsize=14,
            fontweight="bold",
            y=0.99
        )
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[+] Comparison grid saved to: {save_path}")
        plt.close(fig)
