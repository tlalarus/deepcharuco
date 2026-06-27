import json
import os
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


class MiniCharucoDataset(Dataset):
    def __init__(
        self,
        configs,
        labels: str,
        images_folder: str,
        validation: bool = False,
        visualize: bool = False,
    ):
        super().__init__()
        self.configs = configs
        self._images_folder = images_folder
        self._visualize = visualize
        self.stride = int(configs.mini_stride)
        self.sigma = float(configs.mini_heatmap_sigma)
        self.n_corners = int(configs.n_ids)

        with open(labels, "r") as f:
            annotations = json.load(f)
        self.labels = annotations["images"]

        seed = 42 if validation else None
        self.transform = Transformation(configs, negative_p=0.05, refinenet=False, seed=seed)

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

            _draw_gaussian(heatmap[k], xh, yh, self.sigma)

            xi = int(np.floor(xh))
            yi = int(np.floor(yh))
            xi = np.clip(xi, 0, out_w - 1)
            yi = np.clip(yi, 0, out_h - 1)

            offset[0, yi, xi] = xh - xi
            offset[1, yi, xi] = yh - yi
            offset_mask[0, yi, xi] = 1.0

        return heatmap, offset, offset_mask

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        label = self.labels[idx]
        image = cv2.imread(os.path.join(self._images_folder, label["file_name"]), cv2.IMREAD_COLOR)

        transformed = self.transform(image)
        image = transformed["image"]
        keypoints = np.asarray(transformed["keypoints"], dtype=np.float32)
        keypoint_ids = np.asarray(transformed["ids"], dtype=np.int64)
        isnegative = bool(transformed["isnegative"])

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
        return sample

    def __len__(self) -> int:
        return len(self.labels)
