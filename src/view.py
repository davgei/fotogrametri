"""
Interactive 3D viewer for the sparse reconstruction using Plotly.
Opens in the default browser. Supports rotation, zoom, and pan.

Usage:
    py -3.14 src/view.py [--output_dir output]
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pycolmap


def find_best_model(sparse_dir: Path) -> pycolmap.Reconstruction:
    best: pycolmap.Reconstruction | None = None
    for model_dir in sorted(sparse_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        try:
            rec = pycolmap.Reconstruction(str(model_dir))
        except Exception:
            continue
        if best is None or rec.num_reg_images() > best.num_reg_images():
            best = rec
    if best is None:
        raise FileNotFoundError(f"No valid COLMAP models found in '{sparse_dir}'.")
    return best


def camera_centers(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    centers = []
    for image in reconstruction.images.values():
        R = image.cam_from_world().rotation.matrix()
        t = image.cam_from_world().translation
        centers.append(-R.T @ t)
    return np.array(centers)


def main() -> None:
    parser = argparse.ArgumentParser(description="View sparse reconstruction in browser")
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    sparse_dir = args.output_dir / "sparse"
    if not sparse_dir.exists():
        raise FileNotFoundError(
            f"No reconstruction at '{sparse_dir}'. Run reconstruct.py first."
        )

    rec = find_best_model(sparse_dir)
    print(f"Loaded reconstruction: {rec.num_reg_images()} images, {rec.num_points3D()} points")

    pts = rec.points3D
    xyz = np.array([p.xyz for p in pts.values()])
    rgb = np.array([p.color for p in pts.values()])
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]

    point_cloud = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=colors, opacity=0.8),
        name="3D points",
    )

    traces = [point_cloud]

    centers = camera_centers(rec)
    if len(centers):
        traces.append(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode="markers+text",
            marker=dict(size=5, color="red", symbol="diamond"),
            text=[img.name for img in rec.images.values()],
            textposition="top center",
            name="Cameras",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Sparse reconstruction — {rec.num_points3D()} points, {rec.num_reg_images()} cameras",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    html_path = args.output_dir / "reconstruction.html"
    fig.write_html(str(html_path))
    print(f"Saved to '{html_path}' — opening in browser...")
    fig.show()


if __name__ == "__main__":
    main()
