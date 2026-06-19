"""
Full photogrammetry pipeline: sparse SfM + dense MVS (patch match stereo).

Two feature backends for the sparse step:
  --features sift   classical SIFT (default, CPU)
  --features deep   SuperPoint + LightGlue via hloc (GPU, more robust on hard images)

Two dense backends:
  --dense_method colmap    pyColmap patch match stereo -> dense point cloud (default)
  --dense_method openmvs   OpenMVS -> dense point cloud + textured mesh

Usage:
    py -3.14 src/reconstruct.py [--image_dir images] [--output_dir output]
                                [--features sift|deep]
                                [--matching sequential|exhaustive] [--overlap 10]
                                [--no_dense] [--dense_method colmap|openmvs]
                                [--openmvs_bin <dir>]

Dense step requires CUDA (GPU). Use --no_dense to skip it (e.g. on CPU-only machines).
The deep backend requires hloc + torch + GPU:
    pip install git+https://github.com/cvg/Hierarchical-Localization.git
The openmvs dense backend requires the OpenMVS binaries on PATH (or via --openmvs_bin).
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


def write_sequential_pairs(image_names: list[str], pairs_path: Path, overlap: int) -> None:
    names = sorted(image_names)
    lines: list[str] = []
    for i in range(len(names)):
        for j in range(i + 1, min(i + 1 + overlap, len(names))):
            lines.append(f"{names[i]} {names[j]}")
    pairs_path.write_text("\n".join(lines))


def run_sfm_deep(
    image_dir: Path, output_dir: Path, sparse_dir: Path, matching: str, overlap: int
) -> tuple["pycolmap.Reconstruction", Path] | tuple[None, None]:
    from hloc import extract_features as hloc_extract
    from hloc import match_features as hloc_match
    from hloc import pairs_from_exhaustive
    from hloc import reconstruction as hloc_reconstruction

    image_names = [p.name for p in find_images(image_dir)]
    feature_conf = hloc_extract.confs["superpoint_aachen"]
    matcher_conf = hloc_match.confs["superpoint+lightglue"]

    print("  Extracting SuperPoint features...")
    feature_path = hloc_extract.main(feature_conf, image_dir, output_dir)

    pairs_path = output_dir / "pairs.txt"
    if matching == "sequential":
        write_sequential_pairs(image_names, pairs_path, overlap)
    else:
        pairs_from_exhaustive.main(pairs_path, image_list=image_names)

    print("  Matching with LightGlue...")
    match_path = hloc_match.main(matcher_conf, pairs_path, feature_conf["output"], output_dir)

    print("  Running SfM reconstruction (hloc + COLMAP)...")
    model_dir = sparse_dir / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = hloc_reconstruction.main(model_dir, image_dir, pairs_path, feature_path, match_path)
    if model is None:
        return None, None
    return model, model_dir


def undistort(best_model_dir: Path, image_dir: Path, dense_dir: Path) -> None:
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True)

    print("  Undistorting images...")
    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=best_model_dir,
        image_path=image_dir,
    )


def run_dense_colmap(dense_dir: Path) -> Path:
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
    features: str = "sift",
    matching: str = "sequential",
    overlap: int = 10,
    dense: bool = True,
    dense_method: str = "colmap",
    openmvs_bin: Path | None = None,
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

    if features == "deep":
        print(f"Sparse SfM with deep features (SuperPoint + LightGlue, {matching})...")
        best, best_model_dir = run_sfm_deep(image_dir, output_dir, sparse_dir, matching, overlap)
    else:
        print("Step 1/3 — Feature extraction (SIFT)...")
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

    print(f"\nDense reconstruction (MVS, backend={dense_method})...")
    dense_dir = output_dir / "dense"
    undistort(best_model_dir, image_dir, dense_dir)

    if dense_method == "openmvs":
        from src.dense_openmvs import run_dense_openmvs

        result = run_dense_openmvs(dense_dir, output_dir, bin_dir=openmvs_bin)
        print(
            f"\nDone."
            f"\n  Dense point cloud : {result.dense_ply}"
            f"\n  Textured mesh     : {result.textured_mesh}"
        )
    else:
        fused_ply = run_dense_colmap(dense_dir)
        print(f"\nDone. Dense point cloud: {fused_ply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Photogrammetry pipeline via pyColmap")
    parser.add_argument("--image_dir", type=Path, default=Path("images"))
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--features",
        choices=["sift", "deep"],
        default="sift",
        help="sift = classical (CPU); deep = SuperPoint + LightGlue via hloc (GPU)",
    )
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
    parser.add_argument(
        "--dense_method",
        choices=["colmap", "openmvs"],
        default="colmap",
        help="colmap = pyColmap patch match stereo; openmvs = OpenMVS dense + textured mesh",
    )
    parser.add_argument(
        "--openmvs_bin",
        type=Path,
        default=None,
        help="Directory containing OpenMVS binaries (if not on PATH)",
    )
    args = parser.parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: '{args.image_dir}'")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_reconstruction(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        features=args.features,
        matching=args.matching,
        overlap=args.overlap,
        dense=not args.no_dense,
        dense_method=args.dense_method,
        openmvs_bin=args.openmvs_bin,
    )


if __name__ == "__main__":
    main()
