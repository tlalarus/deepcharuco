import os
import socket
import sys

import torch
from torch.utils.data import DataLoader

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import configs
from data import CharucoDataset
from data_mini import MiniCharucoDataset
from models.net import lModel, dcModel
from models.mini_deepcharuco import MiniDeepCharuco, MiniLossConfig, lMiniModel
import pytorch_lightning as pl


def print_runtime_diagnostics():
    print(f"[env] hostname: {socket.gethostname()}")
    print(f"[env] python: {sys.executable}")
    print(f"[env] torch: {torch.__version__}")
    print(f"[env] torch_file: {torch.__file__}")
    print(f"[env] torch_cuda: {torch.version.cuda}")
    print(f"[env] cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        arch_list = torch.cuda.get_arch_list()
        capability = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)
        device_sm = f"sm_{capability[0]}{capability[1]}"
        print(f"[env] cuda_device: {device_name}")
        print(f"[env] cuda_device_sm: {device_sm}")
        print(f"[env] torch_arch_list: {arch_list}")
        strict_check = os.getenv("DEEPCHARUCO_STRICT_CUDA_CHECK", "1") == "1"
        if strict_check and device_sm not in arch_list:
            raise RuntimeError(
                f"CUDA arch mismatch: device {device_sm}, torch supports {arch_list}. "
                f"Check active interpreter/container and reinstall matching torch build."
            )


if __name__ == '__main__':
    print_runtime_diagnostics()
    config = load_configuration(configs.CONFIG_PATH)
    model_type = config.model_type.lower().strip()
    num_workers = int(os.getenv("DEEPCHARUCO_NUM_WORKERS", config.num_workers))
    prefetch_factor = int(os.getenv("DEEPCHARUCO_PREFETCH_FACTOR", "2"))
    pin_memory = torch.cuda.is_available()
    trainer_accelerator = os.getenv("DEEPCHARUCO_ACCELERATOR", "auto")
    trainer_devices_env = os.getenv("DEEPCHARUCO_DEVICES", "auto")
    trainer_devices = int(trainer_devices_env) if trainer_devices_env != "auto" else "auto"
    max_epochs = int(os.getenv("DEEPCHARUCO_MAX_EPOCHS", "100"))
    limit_train_batches = float(os.getenv("DEEPCHARUCO_LIMIT_TRAIN_BATCHES", "1.0"))
    limit_val_batches = float(os.getenv("DEEPCHARUCO_LIMIT_VAL_BATCHES", "1.0"))
    print(f"[env] trainer_accelerator: {trainer_accelerator}")
    print(f"[env] trainer_devices: {trainer_devices}")
    print(f"[env] max_epochs: {max_epochs}")
    print(f"[env] limit_train_batches: {limit_train_batches}")
    print(f"[env] limit_val_batches: {limit_val_batches}")
    if trainer_accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU was forced (DEEPCHARUCO_ACCELERATOR=gpu), "
            "but torch.cuda.is_available() is False."
        )

    if model_type == "mini_deepcharuco":
        dataset = MiniCharucoDataset(config,
                                     config.train_labels,
                                     config.train_images,
                                     visualize=False,
                                     validation=False)

        dataset_val = MiniCharucoDataset(config,
                                         config.val_labels,
                                         config.val_images,
                                         visualize=False,
                                         validation=True)
    elif model_type == "deepcharuco":
        dataset = CharucoDataset(config,
                                 config.train_labels,
                                 config.train_images,
                                 visualize=False,
                                 validation=False)

        dataset_val = CharucoDataset(config,
                                     config.val_labels,
                                     config.val_images,
                                     visualize=False,
                                     validation=True)
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")

    train_loader_kwargs = dict(
        batch_size=config.bs_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader_kwargs = dict(
        batch_size=config.bs_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(dataset, **train_loader_kwargs)
    val_loader = DataLoader(dataset_val, **val_loader_kwargs)

    if model_type == "mini_deepcharuco":
        model = MiniDeepCharuco(
            num_corners=config.n_ids,
            in_channels=1,
            backbone=config.mini_backbone
        )
        loss_cfg = MiniLossConfig(lambda_offset=config.mini_lambda_offset)
        train_model = lMiniModel(model=model,
                                 loss_config=loss_cfg,
                                 lr=config.mini_learning_rate)
        logger = TensorBoardLogger("tb_logs", name="mini_deepcharuco")
        checkpoint_callback = ModelCheckpoint(dirpath="tb_logs/ckpts_mini_deepcharuco/",
                                              save_top_k=10,
                                              monitor="val_loss")
    else:
        model = dcModel(n_ids=config.n_ids)
        train_model = lModel(model)
        logger = TensorBoardLogger("tb_logs", name="deepcharuco")
        checkpoint_callback = ModelCheckpoint(dirpath="tb_logs/ckpts_deepcharuco/",
                                              save_top_k=10,
                                              monitor="val_loss")
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=30,
        min_delta=0.02,
        verbose=True
    )
    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger,
                         accelerator=trainer_accelerator, devices=trainer_devices,
                         limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         callbacks=[checkpoint_callback, early_stopping]) #,
                         # resume_from_checkpoint='./reference/epoch=44-step=83205.ckpt')

    # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(train_model, train_dataloaders=train_loader)
    # print(lr_finder.results)
    # print(lr_finder.suggestion())
    # assert False

    trainer.fit(train_model, train_loader, val_loader)
