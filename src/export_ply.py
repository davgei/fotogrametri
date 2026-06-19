"""
Export the best sparse reconstruction to a .ply file for viewing in
CloudCompare, MeshLab, or any other point cloud viewer.

Usage:
    py -3.14 src/export_ply.py [--output_dir output]
"""

import argparse
from pathlib import Path

import pycolmap


def find_best_model(sparse_dir: Path) -> tuple[pycolmap.Reconstruction, Path]:
    best: pycolmap.Reconstruction | None = None
    best_dir: Path | None = None
    for model_dir in sorted(sparse_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        try:
            rec = pycolmap.Reconstruction(str(model_dir))
        except Exception:
            continue
        if best is None or rec.num_reg_images() > best.num_reg_images():
            best = rec
            best_dir = model_dir
    if best is None:
        raise FileNotFoundError(f"No valid COLMAP models found in '{sparse_dir}'.")
    return best, best_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sparse reconstruction to PLY")
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    sparse_dir = args.output_dir / "sparse"
    if not sparse_dir.exists():
        raise FileNotFoundError(
            f"No reconstruction found at '{sparse_dir}'. Run reconstruct.py first."
        )

    reconstruction, model_dir = find_best_model(sparse_dir)
    ply_path = args.output_dir / "pointcloud.ply"
    reconstruction.export_PLY(str(ply_path))

    print(f"Exported {reconstruction.num_points3D()} points to '{ply_path}'.")
    print(f"  Registered images : {reconstruction.num_reg_images()}")
    print(f"\nOpen '{ply_path}' in CloudCompare or MeshLab to view the point cloud.")
    print("Download CloudCompare: https://www.cloudcompare.org/release/")


if __name__ == "__main__":
    main()
