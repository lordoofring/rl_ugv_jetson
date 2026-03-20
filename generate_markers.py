"""Generate printable ArUco marker images.

Usage:
    python generate_markers.py              # generates markers 0-8 (default)
    python generate_markers.py --ids 0 1 5  # specific IDs only
    python generate_markers.py --size 200   # larger images (pixels)

Output: markers/ directory with individual PNGs and a combined grid sheet.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate ArUco marker images")
    parser.add_argument("--ids", type=int, nargs="+", default=list(range(9)),
                        help="Marker IDs to generate (default: 0-8)")
    parser.add_argument("--size", type=int, default=150,
                        help="Marker image size in pixels (default: 150)")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50",
                        help="ArUco dictionary name")
    parser.add_argument("--outdir", type=str, default="markers",
                        help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    dict_id = getattr(cv2.aruco, args.dict)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    margin = 30  # white border around each marker
    tile_size = args.size + 2 * margin

    images = []
    for mid in args.ids:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, args.size)
        # Add white border + label
        bordered = np.full((tile_size, tile_size), 255, dtype=np.uint8)
        bordered[margin:margin + args.size, margin:margin + args.size] = marker_img
        # Label
        cv2.putText(bordered, f"ID {mid}", (margin, tile_size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

        path = outdir / f"marker_{mid}.png"
        cv2.imwrite(str(path), bordered)
        images.append(bordered)
        print(f"Saved {path}")

    # Combined sheet (3 columns)
    cols = 3
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * tile_size, cols * tile_size), 255, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        sheet[r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size] = img

    sheet_path = outdir / "all_markers.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"\nCombined sheet saved to {sheet_path}")
    print(f"Print at actual size – each marker should be {args.size} pixels "
          f"(verify physical size matches marker_size in config.yaml)")


if __name__ == "__main__":
    main()
