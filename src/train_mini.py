import os
import socket
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import configs
from configs import load_configuration
from data_mini import MiniCharucoDataset
from models.mini_deepcharuco import MiniDeepCharuco, MiniLossConfig, lMiniModel
from real_validation import RealValidationCallback


def print_runtime_diagnostics() -> None:
    print(f"[env] hostname: {socket.gethostname()}")
    print(f"[env] python: {sys.executable}")
    print(f"[env] torch: {torch.__version__}")
    print(f"[env] torch_file: {torch.__file__}")
    print(f"[env] torch_cuda: {torch.version.cuda}")
    print(f"[env] cuda_available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    print_runtime_diagnostics()

    config = load_configuration(configs.CONFIG_PATH)
    num_workers = int(os.getenv("DEEPCHARUCO_NUM_WORKERS", config.num_workers))
    prefetch_factor = int(os.getenv("DEEPCHARUCO_PREFETCH_FACTOR", "2"))
    pin_memory = torch.cuda.is_available()

    trainer_accelerator = os.getenv("DEEPCHARUCO_ACCELERATOR", "auto")
    trainer_devices_env = os.getenv("DEEPCHARUCO_DEVICES", "auto")
    trainer_devices = int(trainer_devices_env) if trainer_devices_env != "auto" else "auto"

    train_ds = MiniCharucoDataset(
        config,
        config.train_labels,
        config.train_images,
        validation=False,
        visualize=False,
    )
    val_ds = MiniCharucoDataset(
        config,
        config.val_labels,
        config.val_images,
        validation=True,
        visualize=False,
    )

    train_loader_kwargs = {
        "batch_size": config.bs_train,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    val_loader_kwargs = {
        "batch_size": config.bs_val,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    real_val_loader = None
    real_val_callback = None
    if config.use_real_val:
        if not config.real_val_dir:
            raise ValueError(
                "use_real_val=True requires real_val_dir to be set in the config."
            )
        manifest_path = os.path.join(config.real_val_dir, "manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Real validation manifest not found at {manifest_path}. "
                "Run src/prepare_real_val.py to generate the dataset."
            )

        real_ds = MiniCharucoDataset(
            config,
            manifest_path,
            config.real_val_dir,
            validation=True,
            visualize=False,
            real_val=True,
            return_raw=True,
        )
        real_loader_kwargs = {
            "batch_size": config.real_val_batch_size,
            "shuffle": False,
            "num_workers": config.real_val_num_workers,
            "pin_memory": pin_memory,
        }
        if config.real_val_num_workers > 0:
            real_loader_kwargs["prefetch_factor"] = prefetch_factor
        real_val_loader = DataLoader(real_ds, **real_loader_kwargs)
        real_val_callback = RealValidationCallback(
            real_val_loader=real_val_loader,
            real_every=config.real_val_every,
            stride=config.mini_stride,
        )

    model = MiniDeepCharuco(
        num_corners=config.n_ids,
        in_channels=1,
        backbone=config.mini_backbone,
    )
    loss_cfg = MiniLossConfig(lambda_offset=config.mini_lambda_offset)
    lit_model = lMiniModel(model=model, loss_config=loss_cfg, lr=config.mini_learning_rate)

    logger = TensorBoardLogger("tb_logs", name="mini_deepcharuco")
    monitor_name = "real_val_mean_error_px" if config.use_real_val else "val_loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath="tb_logs/ckpts_mini_deepcharuco/",
        filename="{epoch:03d}-{step:07d}-" + monitor_name + "={" + monitor_name + ":.6f}",
        save_top_k=10,
        monitor=monitor_name,
        mode="min",
    )

    callbacks = [checkpoint_callback]
    if real_val_callback is not None:
        callbacks.append(real_val_callback)

    trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,
        accelerator=trainer_accelerator,
        devices=trainer_devices,
        callbacks=callbacks,
        check_val_every_n_epoch=config.val_every,
    )
    trainer.fit(lit_model, train_loader, val_loader)
