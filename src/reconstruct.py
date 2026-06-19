"""
Full photogrammetry pipeline: sparse SfM + dense MVS (patch match stereo).

Usage:
    py -3.14 src/reconstruct.py [--image_dir images] [--output_dir output]
                                [--matching sequential|exhaustive] [--overlap 10]
                                [--no_dense]

Dense step requires CUDA (GPU). Use --no_dense to skip it (e.g. on CPU-only machines).
"""

import argparse
import shutil
from pathlib import Path

import pycolmap


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def extract_features(database_path: Path, image_dir: Path) -> None:
    opts = pycolmap.FeatureExtractionOptions()
    opts.sift.max_num_features = 8192
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        extraction_options=opts,
    )


def match_features(database_path: Path, method: str, overlap: int) -> None:
    if method == "sequential":
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = overlap
        pycolmap.match_sequential(database_path=database_path, pairing_options=pairing)
    else:
        pycolmap.match_exhaustive(database_path=database_path)


def run_sfm(
    database_path: Path, image_dir: Path, sparse_dir: Path
) -> tuple[pycolmap.Reconstruction, Path] | tuple[None, None]:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    models = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_dir,
    )
    if not models:
        return None, None
    best_id, best = max(models.items(), key=lambda kv: kv[1].num_reg_images())
    best_dir = sparse_dir / str(best_id)
    return best, best_dir


def run_dense(best_model_dir: Path, image_dir: Path, dense_dir: Path) -> Path:
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True)

    print("  Undistorting images...")
    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=best_model_dir,
        image_path=image_dir,
    )

    print("  Running patch match stereo (GPU required)...")
    pycolmap.patch_match_stereo(workspace_path=dense_dir)

    fused_ply = dense_dir / "fused.ply"
    print("  Fusing depth maps into point cloud...")
    pycolmap.stereo_fusion(
        output_path=fused_ply,
        workspace_path=dense_dir,
        output_type="ply",
    )
    return fused_ply


def run_reconstruction(
    image_dir: Path,
    output_dir: Path,
    matching: str = "sequential",
    overlap: int = 10,
    dense: bool = True,
) -> None:
    images = find_images(image_dir)
    if len(images) < 2:
        raise ValueError(f"At least 2 images required. Found {len(images)} in '{image_dir}'.")
    print(f"Found {len(images)} images.")

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"

    if database_path.exists():
        database_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)

    print("Step 1/3 — Feature extraction...")
    extract_features(database_path, image_dir)

    print(f"Step 2/3 — Feature matching ({matching}, overlap={overlap})...")
    match_features(database_path, method=matching, overlap=overlap)

    print("Step 3/3 — Sparse SfM reconstruction...")
    best, best_model_dir = run_sfm(database_path, image_dir, sparse_dir)

    if best is None:
        print("SfM failed: no reconstruction produced.")
        print("Check that images have sufficient overlap and visual similarity.")
        return

    print(
        f"\nSparse reconstruction:"
        f"\n  Registered images : {best.num_reg_images()} / {len(images)}"
        f"\n  3D points         : {best.num_points3D()}"
    )

    if not dense:
        print(f"\nDone. Sparse model saved to: {sparse_dir}")
        return

    print("\nDense reconstruction (MVS)...")
    dense_dir = output_dir / "dense"
    fused_ply = run_dense(best_model_dir, image_dir, dense_dir)
    print(f"\nDone. Dense point cloud: {fused_ply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Photogrammetry pipeline via pyColmap")
    parser.add_argument("--image_dir", type=Path, default=Path("images"))
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--matching",
        choices=["sequential", "exhaustive"],
        default="sequential",
        help="sequential = fast (ordered images); exhaustive = slow (unordered)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Number of neighboring images to match (sequential only, default: 10)",
    )
    parser.add_argument(
        "--no_dense",
        action="store_true",
        help="Skip dense reconstruction (faster, no GPU required)",
    )
    args = parser.parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: '{args.image_dir}'")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_reconstruction(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        matching=args.matching,
        overlap=args.overlap,
        dense=not args.no_dense,
    )


if __name__ == "__main__":
    main()
