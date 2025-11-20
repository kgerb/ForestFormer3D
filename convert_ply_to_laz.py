#!/usr/bin/env python3
"""Convert a prediction PLY file (x y z semantic instance) into a LAZ file.

The script keeps only the essentials:
    * reads ASCII PLY (five columns expected)
    * optionally copies RGB from an original LAZ (otherwise uses a fixed palette)
    * writes LAS/LAZ with semantic/instance extra dimensions

Usage
-----
python convert_ply_to_laz.py \
    --input-ply path/to/predictions.ply \
    --output-laz path/to/output.laz \
    [--color-source path/to/original.laz]
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

try:
    import laspy
except ImportError as exc:  # pragma: no cover - user must install manually
    raise SystemExit(
        "laspy is required for LAZ export. Install with `pip install laspy`."
    ) from exc


def read_ascii_ply(ply_path: str) -> np.ndarray:
    """Load an ASCII PLY with five columns (x y z semantic instance)."""
    with open(ply_path, "r", encoding="utf-8") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected EOF before end_header")
            header.append(line.strip())
            if line.strip() == "end_header":
                break

        vertex_count = None
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                break
        if vertex_count is None:
            raise ValueError("PLY header missing `element vertex` entry")

        data = []
        for idx in range(vertex_count):
            row = handle.readline()
            if not row:
                raise ValueError("Unexpected EOF while reading vertices")
            parts = row.strip().split()
            if len(parts) < 5:
                raise ValueError(
                    f"Vertex #{idx} has {len(parts)} values, expected at least 5"
                )
            data.append([float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3]), int(parts[4])])

    return np.asarray(data, dtype=np.float64)


def rgb_from_semantics(semantics: np.ndarray) -> np.ndarray:
    """Return simple 16-bit RGB palette for semantic IDs."""
    palette = {
        0: (139, 90, 43),   # ground
        1: (101, 67, 33),   # wood
        2: (34, 139, 34),   # leaf
    }
    rgb = np.zeros((semantics.size, 3), dtype=np.uint16)
    for sem_id, color in palette.items():
        mask = semantics == sem_id
        if mask.any():
            rgb[mask] = np.uint16(color) * 256
    return rgb


def rgb_from_reference_laz(
    laz_path: str, target_coords: np.ndarray, tolerance: float = 1e-5
) provides np.ndarray:
    """Fetch RGB by nearest-neighbour lookup against an existing LAZ."""
    reference = laspy.read(laz_path)
    source_xyz = np.column_stack((reference.x, reference.y, reference.z))
    source_rgb = np.column_stack(
        (
            reference.red.astype(np.uint16),
            reference.green.astype(np.uint16),
            reference.blue.astype(np.uint16),
        )
    )

    try:
        from scipy.spatial import cKDTree  # lazy import
    except ImportError as exc:  # pragma: no cover - optional speedup
        raise SystemExit(
            "scipy is required for --color-source matching. Install with `pip install scipy`."
        ) from exc

    tree = cKDTree(source_xyz)
    distances, indices = tree.query(target_coords, k=1)
    too_far = distances > tolerance
    if np.any(too_far):
        print(
            f"[WARN] {np.count_nonzero(too_far)} points had NN distance > {tolerance:.1e}; "
            "using their nearest colors anyway."
        )
    return source_rgb[indices]


def write_laz(
    points: np.ndarray,
    output_path: str,
    rgb: Optional[np.ndarray] = None,
) -> None:
    """Write points + rgb + semantic/instance extras to output_path."""
    las = laspy.create(point_format=2, file_version="1.2")
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    semantics = points[:, 3].astype(np.int32)
    instances = points[:, 4].astype(np.int32)

    if rgb is None:
        rgb = rgb_from_semantics(semantics)
    if rgb.shape != (points.shape[0], 3):
        raise ValueError("RGB array must have shape (N, 3)")

    las.red = rgb[:, 0]
    las.green = rgb[:, 1]
    las.blue = rgb[:, 2]

    las.add_extra_dim(laspy.ExtraBytesParams(name="semantic_pred", type=np.int32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="instance_pred", type=np.int32))
    las.semantic_pred = semantics
    las.instance_pred = instances

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    las.write(output_path)
    print(f"[OK] Wrote {points.shape[0]} points to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True, help="Prediction PLY with x y z semantic instance.")
    parser.add_argument("--output-laz", required=True, help="Destination LAZ path.")
    parser.add_argument(
        "--color-source",
        help="Optional reference LAZ to copy RGB from (requires scipy). If omitted, a fixed palette is used.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=1e-5,
        help="Max distance (in meters) when matching to --color-source (default: 1e-5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = read_ascii_ply(args.input_ply)

    rgb = None
    if args.color_source:
        print(f"[INFO] Copying RGB from {args.color_source}")
        rgb = rgb_from_reference_laz(args.color_source, points[:, :3], args.match_tolerance)

    write_laz(points, args.output_laz, rgb)


if __name__ == "__main__":
    main()
