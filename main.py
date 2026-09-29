"""
main.py
=======
Command-line interface and demonstration script for SVD Image Compression.

Usage:
    python main.py
    python main.py --image assets/horse.jpg --k 5 15 30 50 100 200
    python main.py --help
"""

import argparse
import os
import sys
from src.svd_compressor import SVDImageCompressor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-Rank Image Compression via Singular Value Decomposition (SVD) with NumPy"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "assets", "horse.jpg"),
        help="Path to the input image (default: assets/horse.jpg)"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 15, 30, 50, 100, 200],
        help="List of singular value ranks (k) to compute and compare (default: 5 15 30 50 100 200)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "outputs"),
        help="Directory to save generated comparison plots and compressed images"
    )
    parser.add_argument(
        "--save-assets",
        action="store_true",
        help="Also update preview plots in assets/ directory for GitHub README"
    )
    return parser.parse_args()


def print_banner(compressor: SVDImageCompressor) -> None:
    mode = "RGB Color" if compressor.is_color else "Grayscale"
    print("=" * 80)
    print("  SINGULAR VALUE DECOMPOSITION (SVD) - LOW-RANK IMAGE COMPRESSION ENGINE")
    print("=" * 80)
    print(f"  Input Image       : {os.path.basename(compressor.image_path)}")
    print(f"  Resolution (HxW)  : {compressor.height} x {compressor.width}")
    print(f"  Channels          : {compressor.num_channels} ({mode})")
    print(f"  Maximum Rank      : {compressor.max_rank} singular values")
    print(f"  Total Data Points : {compressor.height * compressor.width * compressor.num_channels:,} floats")
    print("-" * 80)


def print_metrics_table(metrics_list: list) -> None:
    header = f"{'Rank (k)':<10} | {'Stored Elements':<18} | {'Compression':<13} | {'Space Saved':<13} | {'Energy (%)':<12} | {'Frobenius Error':<16}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        row = (
            f"{m['k']:<10} | "
            f"{m['svd_elements']:<18,d} | "
            f"{m['compression_ratio']:<13.2f}x | "
            f"{m['space_savings_pct']:<12.1f}% | "
            f"{m['energy_preserved_pct']:<11.2f}% | "
            f"{m['relative_frobenius_error']:<16.4f}"
        )
        print(row)
    print("-" * len(header))


def main() -> None:
    args = parse_arguments()

    if not os.path.exists(args.image):
        print(f"[!] Error: Image not found at '{args.image}'. Please provide a valid path.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1/4] Loading image and computing batched SVD...")
    compressor = SVDImageCompressor(args.image)
    print_banner(compressor)

    # Filter k to be within valid range
    valid_k = sorted(list(set([k for k in args.k if 1 <= k <= compressor.max_rank])))
    if not valid_k:
        valid_k = [10, 20, 50]

    print("\n[2/4] Computing rank-k approximations and compression metrics...")
    metrics_list = []
    for k in valid_k:
        metrics = compressor.compute_metrics(k)
        metrics_list.append(metrics)
    print_metrics_table(metrics_list)

    print("\n[3/4] Generating visual comparison grid...")
    grid_out = os.path.join(args.output_dir, "svd_compression_grid.png")
    compressor.plot_comparison_grid(valid_k, save_path=grid_out)

    print("\n[4/4] Generating singular values decay spectrum...")
    spectrum_out = os.path.join(args.output_dir, "singular_values_spectrum.png")
    compressor.plot_singular_values(save_path=spectrum_out)

    if args.save_assets:
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        compressor.plot_comparison_grid(valid_k, save_path=os.path.join(assets_dir, "svd_compression_grid.png"))
        compressor.plot_singular_values(save_path=os.path.join(assets_dir, "singular_values_spectrum.png"))

    print("\n" + "=" * 80)
    print(f"  SUCCESS! Outputs generated in: {args.output_dir}")
    print(f"  - Comparison Grid : {grid_out}")
    print(f"  - Spectrum Plot   : {spectrum_out}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
