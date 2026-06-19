"""
Dense reconstruction + textured mesh via OpenMVS.

Takes a COLMAP workspace produced by pycolmap.undistort_images (undistorted
images + sparse model) and runs the OpenMVS CLI pipeline:

    InterfaceCOLMAP   COLMAP workspace -> scene.mvs
    DensifyPointCloud scene.mvs        -> dense point cloud
    ReconstructMesh   scene_dense.mvs  -> mesh
    TextureMesh       *_mesh.mvs       -> textured mesh

OpenMVS is a set of C++ command-line tools, not a Python package. The binaries
must be available on PATH, or pass their directory via bin_dir / --openmvs_bin.
"""

import shutil
import subprocess
from pathlib import Path


OPENMVS_TOOLS = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh"]


class OpenMVSResult:
    def __init__(self, dense_ply: Path, mesh_ply: Path, textured_mesh: Path) -> None:
        self.dense_ply = dense_ply
        self.mesh_ply = mesh_ply
        self.textured_mesh = textured_mesh


def _tool_path(name: str, bin_dir: Path | None) -> str:
    return str(bin_dir / name) if bin_dir is not None else name


def verify_tools_available(bin_dir: Path | None) -> None:
    missing = [t for t in OPENMVS_TOOLS if shutil.which(_tool_path(t, bin_dir)) is None]
    if missing:
        raise FileNotFoundError(
            f"OpenMVS tools not found: {missing}. "
            f"Build OpenMVS and put its binaries on PATH, or pass --openmvs_bin <dir>."
        )


def _run(cmd: list[str], cwd: Path) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def run_dense_openmvs(
    dense_dir: Path, output_dir: Path, bin_dir: Path | None = None
) -> OpenMVSResult:
    verify_tools_available(bin_dir)

    mvs_dir = output_dir / "openmvs"
    if mvs_dir.exists():
        shutil.rmtree(mvs_dir)
    mvs_dir.mkdir(parents=True)

    images_dir = dense_dir / "images"

    print("  OpenMVS 1/4 — InterfaceCOLMAP (importing COLMAP workspace)...")
    _run(
        [
            _tool_path("InterfaceCOLMAP", bin_dir),
            "-i", str(dense_dir),
            "-o", "scene.mvs",
            "--image-folder", str(images_dir),
        ],
        cwd=mvs_dir,
    )

    print("  OpenMVS 2/4 — DensifyPointCloud...")
    _run([_tool_path("DensifyPointCloud", bin_dir), "scene.mvs"], cwd=mvs_dir)

    print("  OpenMVS 3/4 — ReconstructMesh...")
    _run([_tool_path("ReconstructMesh", bin_dir), "scene_dense.mvs"], cwd=mvs_dir)

    print("  OpenMVS 4/4 — TextureMesh...")
    _run([_tool_path("TextureMesh", bin_dir), "scene_dense_mesh.mvs"], cwd=mvs_dir)

    return OpenMVSResult(
        dense_ply=mvs_dir / "scene_dense.ply",
        mesh_ply=mvs_dir / "scene_dense_mesh.ply",
        textured_mesh=mvs_dir / "scene_dense_mesh_texture.ply",
    )
