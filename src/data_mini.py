import csv
import json
import os
import warnings
from typing import Tuple

import cv2
import numpy as np
from torch.utils.data.dataset import Dataset

from models.model_utils import pre_bgr_image
from transformations import Transformation


def _draw_gaussian(heatmap: np.ndarray, center_x: float, center_y: float, sigma: float) -> None:
    h, w = heatmap.shape
    radius = max(1, int(3 * sigma))

    x0 = max(0, int(center_x) - radius)
    x1 = min(w - 1, int(center_x) + radius)
    y0 = max(0, int(center_y) - radius)
    y1 = min(h - 1, int(center_y) + radius)

    if x0 > x1 or y0 > y1:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma))
    heatmap[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(
        heatmap[y0 : y1 + 1, x0 : x1 + 1], gaussian
    )


def _find_header_key(fieldnames, candidates):
    if fieldnames is None:
        return None
    normalized = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        candidate = candidate.lower()
        if candidate in normalized:
            return normalized[candidate]
    return None


def _parse_corner_csv(csv_path: str, image_shape: Tuple[int, int]):
    height, width = image_shape
    keypoints = []
    keypoint_ids = []
    valid_mask = []
    out_of_bounds = False

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Corner CSV '{csv_path}' has no header row.")

        x_key = _find_header_key(reader.fieldnames, ["x"])
        y_key = _find_header_key(reader.fieldnames, ["y"])
        id_key = _find_header_key(reader.fieldnames, ["corner_id", "id"])
        valid_key = _find_header_key(reader.fieldnames, ["visible", "valid"])

        if x_key is None or y_key is None:
            raise ValueError(
                f"Corner CSV '{csv_path}' must include 'x' and 'y' columns. Found: {reader.fieldnames}"
            )

        for row_index, row in enumerate(reader):
            row_data = {k.strip().lower(): v for k, v in row.items() if k is not None}
            try:
                x = float(row_data[x_key.strip().lower()])
                y = float(row_data[y_key.strip().lower()])
            except (KeyError, ValueError):
                raise ValueError(
                    f"Invalid x/y value in '{csv_path}' row {row_index + 1}: {row}"
                )

            if x < 0 or x > width - 1 or y < 0 or y > height - 1:
                out_of_bounds = True

            if id_key is not None:
                raw_id = row_data.get(id_key.strip().lower(), "")
                try:
                    corner_id = int(raw_id)
                except ValueError:
                    corner_id = row_index
            else:
                corner_id = row_index

            visible = True
            if valid_key is not None:
                raw_visible = row_data.get(valid_key.strip().lower(), "1")
                visible = str(raw_visible).strip().lower() not in {"0", "false", "no", "n"}

            if not visible:
                continue

            keypoints.append((x, y))
            keypoint_ids.append(corner_id)
            valid_mask.append(1)

    if out_of_bounds:
        warnings.warn(
            f"Corner CSV '{csv_path}' contains coordinates outside the image bounds {image_shape}."
        )

    return (
        np.asarray(keypoints, dtype=np.float32),
        np.asarray(keypoint_ids, dtype=np.int64),
        np.asarray(valid_mask, dtype=np.float32),
    )


class MiniCharucoDataset(Dataset):
    def __init__(
        self,
        configs,
        labels: str,
        images_folder: str,
        validation: bool = False,
        visualize: bool = False,
        real_val: bool = False,
        return_raw: bool = False,
    ):
        super().__init__()
        self.configs = configs
        self._images_folder = images_folder
        self._visualize = visualize
        self.stride = int(configs.mini_stride)
        self.sigma = float(configs.mini_heatmap_sigma)
        self.n_corners = int(configs.n_ids)
        self.real_val = real_val or labels.lower().endswith(".csv")
        self.return_raw = return_raw
        self._manifest_dir = os.path.dirname(labels)

        if self.real_val:
            self.labels = self._load_real_manifest(labels)
        else:
            with open(labels, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            self.labels = annotations["images"]

        if self.real_val:
            self.transform = self._build_validation_transform(configs.input_size)
        else:
            seed = 42 if validation else None
            self.transform = Transformation(configs, negative_p=0.05, refinenet=False, seed=seed)

    def _load_real_manifest(self, manifest_path: str):
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Real validation manifest not found: {manifest_path}")

        rows = []
        with open(manifest_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest file '{manifest_path}' has no header row.")

            image_key = _find_header_key(reader.fieldnames, ["image_path", "image", "image_name"])
            csv_key = _find_header_key(reader.fieldnames, ["csv_path", "corners", "csv"])
            if image_key is None or csv_key is None:
                raise ValueError(
                    f"Manifest '{manifest_path}' must include 'image_path' and 'csv_path' columns."
                )

            for row_index, row in enumerate(reader):
                row_data = {k.strip().lower(): v for k, v in row.items() if k is not None}
                image_path = row_data.get(image_key.strip().lower(), "").strip()
                csv_path = row_data.get(csv_key.strip().lower(), "").strip()
                if not image_path or not csv_path:
                    raise ValueError(
                        f"Manifest row {row_index + 1} missing image_path or csv_path: {row}"
                    )

                image_path = self._resolve_manifest_path(manifest_path, image_path)
                csv_path = self._resolve_manifest_path(manifest_path, csv_path)
                image_id = row_data.get("image_id", os.path.splitext(os.path.basename(image_path))[0])
                width = int(row_data.get("width", 0))
                height = int(row_data.get("height", 0))
                orig_size = (height, width) if width > 0 and height > 0 else None

                rows.append(
                    {
                        "image_path": image_path,
                        "csv_path": csv_path,
                        "image_id": image_id,
                        "orig_size": orig_size,
                    }
                )

        if not rows:
            raise ValueError(f"Real validation manifest '{manifest_path}' contains no entries.")

        return rows

    def _resolve_manifest_path(self, manifest_path: str, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(os.path.dirname(manifest_path), path)

    def _build_validation_transform(self, input_size: Tuple[int, int]):
        import albumentations as A

        width, height = input_size
        return A.Compose(
            [
                A.PadIfNeeded(min_height=height, min_width=width, always_apply=True,
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Resize(height=height, width=width, always_apply=True),
            ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=['ids']),
        )

    def _build_targets(
        self,
        image_shape: Tuple[int, int],
        keypoints: np.ndarray,
        keypoint_ids: np.ndarray,
        isnegative: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image_shape
        out_h = h // self.stride
        out_w = w // self.stride

        heatmap = np.zeros((self.n_corners, out_h, out_w), dtype=np.float32)
        offset = np.zeros((2, out_h, out_w), dtype=np.float32)
        offset_mask = np.zeros((1, out_h, out_w), dtype=np.float32)

        if isnegative:
            return heatmap, offset, offset_mask

        for kp, idx in zip(keypoints, keypoint_ids):
            k = int(idx)
            if k < 0 or k >= self.n_corners:
                continue

            x, y = float(kp[0]), float(kp[1])
            xh = x / self.stride
            yh = y / self.stride

            if xh < 0 or yh < 0 or xh >= out_w or yh >= out_h:
                continue

            xi = int(np.floor(xh))
            yi = int(np.floor(yh))
            xi = np.clip(xi, 0, out_w - 1)
            yi = np.clip(yi, 0, out_h - 1)

            # Heatmap peak and offset supervision must use the same spatial cell.
            _draw_gaussian(heatmap[k], xi, yi, self.sigma)

            offset[0, yi, xi] = xh - xi
            offset[1, yi, xi] = yh - yi
            offset_mask[0, yi, xi] = 1.0

        return heatmap, offset, offset_mask

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        record = self.labels[idx]
        if self.real_val:
            image_path = record["image_path"]
            csv_path = record["csv_path"]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to read real validation image '{image_path}'")

            keypoints, keypoint_ids, valid_mask = _parse_corner_csv(csv_path, image.shape[:2])
            transformed = self.transform(image=image, keypoints=keypoints.tolist(), ids=keypoint_ids.tolist())
            image = transformed["image"]
            keypoints = np.asarray(transformed["keypoints"], dtype=np.float32)
            keypoint_ids = np.asarray(transformed["ids"], dtype=np.int64)
            isnegative = False
        else:
            label = record
            image = cv2.imread(os.path.join(self._images_folder, label["file_name"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(
                    f"Failed to read training/validation image '{label['file_name']}' from {self._images_folder}"
                )
            transformed = self.transform(image)
            image = transformed["image"]
            keypoints = np.asarray(transformed["keypoints"], dtype=np.float32)
            keypoint_ids = np.asarray(transformed["ids"], dtype=np.int64)
            isnegative = bool(transformed["isnegative"])
            valid_mask = np.ones((keypoints.shape[0],), dtype=np.float32)

        heatmap, offset, offset_mask = self._build_targets(
            image_shape=(image.shape[0], image.shape[1]),
            keypoints=keypoints,
            keypoint_ids=keypoint_ids,
            isnegative=isnegative,
        )

        image = pre_bgr_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).astype(np.float32)

        sample = {
            "image": image,
            "heatmap": heatmap,
            "offset": offset,
            "offset_mask": offset_mask,
        }

        if self.real_val and self.return_raw:
            sample.update(
                {
                    "keypoints": keypoints,
                    "ids": keypoint_ids,
                    "valid_mask": valid_mask,
                    "image_id": record.get("image_id", os.path.splitext(os.path.basename(image_path))[0]),
                    "orig_size": record.get("orig_size", tuple(image.shape[:2])),
                }
            )

        return sample

    def __len__(self) -> int:
        return len(self.labels)
