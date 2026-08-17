import csv
import os
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


def decode_heatmap_offset(heatmap: torch.Tensor, offset: torch.Tensor, stride: int = 4) -> torch.Tensor:
    assert heatmap.ndim == 4 and offset.ndim == 4
    batch_size, num_corners, height, width = heatmap.shape
    flat_heatmap = heatmap.view(batch_size, num_corners, -1)
    indices = flat_heatmap.argmax(dim=-1)
    ys = indices // width
    xs = indices % width

    # offset has no corner axis; gather spatial offsets at each corner's argmax location.
    offset_x = offset[:, 0].reshape(batch_size, -1).gather(1, indices)
    offset_y = offset[:, 1].reshape(batch_size, -1).gather(1, indices)

    pred_x = (xs.float() + offset_x) * stride
    pred_y = (ys.float() + offset_y) * stride
    return torch.stack((pred_x, pred_y), dim=-1)


def _unwrap_batch_value(value):
    if isinstance(value, list) or isinstance(value, tuple):
        return value[0]
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return value.squeeze(0)
    return value


class RealValidationCallback(Callback):
    def __init__(
        self,
        real_val_loader,
        real_every: int = 1,
        stride: int = 4,
        metrics_file: str | None = None,
    ):
        super().__init__()
        self.real_val_loader = real_val_loader
        self.real_every = max(1, real_every)
        self.stride = stride
        self.metrics_file = metrics_file

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        if (epoch + 1) % self.real_every != 0:
            return

        self._run_real_validation(trainer, pl_module, epoch)

    def _run_real_validation(self, trainer, pl_module, epoch: int):
        was_training = pl_module.training
        pl_module.eval()
        device = pl_module.device

        image_rows = []
        errors_all = []
        num_points = 0
        num_images = 0

        with torch.inference_mode():
            for batch in self.real_val_loader:
                image_id = None
                if "image_id" in batch:
                    image_id = _unwrap_batch_value(batch["image_id"])
                if image_id is None:
                    image_id = f"image_{num_images}"

                image = batch["image"].to(device)
                pred = pl_module.model(image)
                pred_coords = decode_heatmap_offset(pred["heatmap"], pred["offset"], stride=self.stride)

                gt_keypoints = _unwrap_batch_value(batch["keypoints"])
                ids = _unwrap_batch_value(batch["ids"])
                valid_mask = _unwrap_batch_value(batch["valid_mask"])

                if isinstance(gt_keypoints, torch.Tensor):
                    gt_keypoints = gt_keypoints.cpu().numpy()
                if isinstance(ids, torch.Tensor):
                    ids = ids.cpu().numpy()
                if isinstance(valid_mask, torch.Tensor):
                    valid_mask = valid_mask.cpu().numpy()

                if gt_keypoints is None or ids is None or valid_mask is None:
                    raise ValueError("Real validation batch missing keypoints/ids/valid_mask")

                gt_keypoints = np.asarray(gt_keypoints, dtype=np.float32)
                ids = np.asarray(ids, dtype=np.int64)
                valid_mask = np.asarray(valid_mask, dtype=np.bool_) if valid_mask.size else np.array([], dtype=np.bool_)

                image_errors = []
                valid_corners = int(np.sum(valid_mask))
                for gt, idx, valid in zip(gt_keypoints, ids, valid_mask):
                    if not valid:
                        continue
                    if idx < 0 or idx >= pred_coords.shape[1]:
                        continue
                    pred_pt = pred_coords[0, int(idx)].cpu().numpy()
                    image_errors.append(float(np.linalg.norm(pred_pt - gt)))

                image_errors = np.asarray(image_errors, dtype=np.float32)
                errors_all.extend(image_errors.tolist())
                num_points += image_errors.shape[0]
                num_images += 1

                if image_errors.size > 0:
                    mean_error = float(image_errors.mean())
                    median_error = float(np.median(image_errors))
                    pck_2px = float(np.mean(image_errors <= 2.0))
                    pck_5px = float(np.mean(image_errors <= 5.0))
                else:
                    mean_error = float("nan")
                    median_error = float("nan")
                    pck_2px = 0.0
                    pck_5px = 0.0

                image_rows.append(
                    {
                        "epoch": epoch + 1,
                        "image_id": str(image_id),
                        "num_valid_corners": valid_corners,
                        "num_eval_corners": int(image_errors.shape[0]),
                        "mean_error_px": mean_error,
                        "median_error_px": median_error,
                        "pck_2px": pck_2px,
                        "pck_5px": pck_5px,
                    }
                )

        if errors_all:
            errors_all = np.asarray(errors_all, dtype=np.float32)
            mean_error = float(errors_all.mean())
            median_error = float(np.median(errors_all))
            pck_2px = float(np.mean(errors_all <= 2.0))
            pck_5px = float(np.mean(errors_all <= 5.0))
        else:
            mean_error = float("nan")
            median_error = float("nan")
            pck_2px = 0.0
            pck_5px = 0.0

        metrics = {
            "real_val_mean_error_px": mean_error,
            "real_val_median_error_px": median_error,
            "real_val_pck_2px": pck_2px,
            "real_val_pck_5px": pck_5px,
            "real_val_num_images": num_images,
            "real_val_num_points": num_points,
            "real_val/mean_error_px": mean_error,
            "real_val/median_error_px": median_error,
            "real_val/pck_2px": pck_2px,
            "real_val/pck_5px": pck_5px,
            "real_val/num_images": num_images,
            "real_val/num_points": num_points,
        }

        for key, value in metrics.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                metrics[key] = 0.0

        self._record_metrics(trainer, metrics)
        self._save_image_metrics(trainer, image_rows)

        if was_training:
            pl_module.train()

    def _record_metrics(self, trainer, metrics):
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        for key, value in metrics.items():
            if key.startswith("real_val_"):
                trainer.callback_metrics[key] = torch.tensor(value)

    def _save_image_metrics(self, trainer, image_rows):
        if not self.metrics_file and trainer.logger is not None:
            self.metrics_file = os.path.join(trainer.logger.log_dir, "real_val_per_image.csv")

        if self.metrics_file is None:
            return

        file_path = Path(self.metrics_file)
        write_header = not file_path.exists()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "image_id",
                    "num_valid_corners",
                    "num_eval_corners",
                    "mean_error_px",
                    "median_error_px",
                    "pck_2px",
                    "pck_5px",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(image_rows)
