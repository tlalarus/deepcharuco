from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np
import torch

import configs
from configs import load_configuration
from models.mini_deepcharuco import MiniDeepCharuco, MiniLossConfig, lMiniModel
from models.model_utils import pre_bgr_image


def decode_corners(
    heatmap_logits: torch.Tensor,
    offset: torch.Tensor,
    stride: int,
    conf_threshold: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode model outputs into corner coordinates.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        corners: [N, 3] with x, y, corner_idx
        confidences: [N]
    """
    if heatmap_logits.ndim == 4:
        heatmap_logits = heatmap_logits[0]
    if offset.ndim == 4:
        offset = offset[0]

    heatmap = torch.sigmoid(heatmap_logits)
    k, h, w = heatmap.shape

    corners = []
    confidences = []
    for idx in range(k):
        channel = heatmap[idx]
        flat_idx = torch.argmax(channel)
        yh = torch.div(flat_idx, w, rounding_mode="floor")
        xh = flat_idx % w

        conf = channel[yh, xh].item()
        if conf_threshold is not None and conf < conf_threshold:
            continue

        dx = offset[0, yh, xh].item()
        dy = offset[1, yh, xh].item()

        x = (xh.item() + dx) * stride
        y = (yh.item() + dy) * stride
        corners.append([x, y, idx])
        confidences.append(conf)

    if len(corners) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.asarray(corners, dtype=np.float32), np.asarray(confidences, dtype=np.float32)


def load_mini_model(ckpt_path: str, config, device: str) -> lMiniModel:
    model = MiniDeepCharuco(
        num_corners=config.n_ids,
        in_channels=1,
        backbone=config.mini_backbone,
    )
    loss_cfg = MiniLossConfig(lambda_offset=config.mini_lambda_offset)
    lit = lMiniModel.load_from_checkpoint(
        ckpt_path,
        model=model,
        loss_config=loss_cfg,
        lr=config.mini_learning_rate,
    )
    lit.eval()
    lit.to(device)
    return lit


@torch.no_grad()
def infer_image(
    image_bgr: np.ndarray,
    model: lMiniModel,
    stride: int,
    device: str,
    conf_threshold: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_tensor = torch.tensor(pre_bgr_image(image_gray), device=device, dtype=torch.float32)

    heatmap_logits, offset = model.infer_image(image_tensor)
    corners, confidences = decode_corners(heatmap_logits, offset, stride, conf_threshold)
    return corners, confidences


def draw_corners(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    for x, y, idx in corners:
        cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
        cv2.putText(
            out,
            str(int(idx)),
            (int(round(x)) + 4, int(round(y)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 50, 50),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--conf", default=None, type=float)
    parser.add_argument("--save", default=None, type=str)
    args = parser.parse_args()

    config = load_configuration(configs.CONFIG_PATH)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    model = load_mini_model(args.ckpt, config, device)

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    corners, confs = infer_image(
        image_bgr=image,
        model=model,
        stride=config.mini_stride,
        device=device,
        conf_threshold=args.conf,
    )

    print("corners:")
    print(corners)
    print("confidences:")
    print(confs)

    if args.save:
        out = draw_corners(image, corners)
        cv2.imwrite(args.save, out)
        print(f"Saved visualization to {args.save}")


if __name__ == "__main__":
    main()
