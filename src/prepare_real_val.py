import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_files(directory, extensions):
    files = {}
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            files[path.stem] = path
    return files


def _ensure_output_paths(root_path: Path):
    images_dir = root_path / "images"
    corners_dir = root_path / "corners"
    images_dir.mkdir(parents=True, exist_ok=True)
    corners_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, corners_dir


def _copy_or_symlink(source: Path, destination: Path, mode: str):
    if mode == "symlink":
        if destination.exists():
            destination.unlink()
        os.symlink(source.resolve(), destination)
    else:
        shutil.copy2(source, destination)


def _build_manifest(records, root_dir: Path, manifest_path: Path):
    fieldnames = ["image_path", "csv_path", "image_id", "width", "height", "num_corners"]
    with manifest_path.open("w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def prepare_real_validation(images_dir: Path, csv_dir: Path, output_root: Path, expected_count: int | None, mode: str = "symlink"):
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    if mode not in {"symlink", "copy"}:
        raise ValueError("mode must be 'symlink' or 'copy'")

    image_files = _collect_files(images_dir, IMAGE_EXTENSIONS)
    csv_files = {path.stem: path for path in Path(csv_dir).iterdir() if path.is_file() and path.suffix.lower() == ".csv"}

    missing_images = sorted(set(csv_files) - set(image_files))
    missing_csvs = sorted(set(image_files) - set(csv_files))
    if missing_images or missing_csvs:
        message = []
        if missing_images:
            message.append(f"Missing image files for CSV basenames: {missing_images}")
        if missing_csvs:
            message.append(f"Missing CSV files for image basenames: {missing_csvs}")
        raise ValueError("Real validation files do not match:\n" + "\n".join(message))

    if expected_count is not None and len(image_files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} real validation examples, but found {len(image_files)}."
        )

    images_out, corners_out = _ensure_output_paths(output_root)
    records = []

    for basename in sorted(image_files):
        image_path = image_files[basename]
        csv_path = csv_files[basename]

        dest_image = images_out / image_path.name
        dest_csv = corners_out / csv_path.name
        _copy_or_symlink(image_path, dest_image, mode)
        _copy_or_symlink(csv_path, dest_csv, mode)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image '{image_path}' for manifest generation.")
        height, width = image.shape[:2]

        num_corners = 0
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for _ in reader:
                num_corners += 1

        records.append(
            {
                "image_path": os.path.join("images", image_path.name),
                "csv_path": os.path.join("corners", csv_path.name),
                "image_id": basename,
                "width": width,
                "height": height,
                "num_corners": num_corners,
            }
        )

    manifest_path = output_root / "manifest.csv"
    _build_manifest(records, output_root, manifest_path)
    print(f"Prepared real validation set at: {output_root}")
    print(f"  images: {images_out}")
    print(f"  corners: {corners_out}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare real validation data for mini deepcharuco.")
    parser.add_argument("--images-dir", required=True, help="Directory with raw validation images.")
    parser.add_argument("--csv-dir", required=True, help="Directory with raw corner CSVs.")
    parser.add_argument("--output-root", default="data/mini_deepcharuco/val_real", help="Output root for prepared real validation data.")
    parser.add_argument("--expected-count", type=int, default=None, help="Expected number of real validation examples.")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink", help="Copy or symlink files into the real validation directory.")
    args = parser.parse_args()

    prepare_real_validation(
        images_dir=Path(args.images_dir),
        csv_dir=Path(args.csv_dir),
        output_root=Path(args.output_root),
        expected_count=args.expected_count,
        mode=args.mode,
    )
