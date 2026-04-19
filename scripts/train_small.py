"""
Small-subset training script for quick sanity / regularization experiments.

Trains on a fixed number of samples so you can check whether val NLL follows
train loss without waiting for a full run.

Usage (from project root):
    python scripts/train_small.py --n_train 1000 --n_val 100

Arguments:
    --n_train       Exact number of training samples    (default 1000)
    --n_val         Exact number of validation samples  (default 100)
    --epochs        Number of training epochs           (default 100)
    --weight_decay  AdamW weight decay                  (default 1e-4)
    --dropout       Dropout probability (0 = off)       (default 0.0)
    --batch_size    Mini-batch size                     (default 32)
    --name          Run name for checkpoint/logging     (default small_test)
    --val_every     Check val every N epochs            (default 5)
    --lr            Learning rate                       (default 2e-4)
"""

import argparse
import sys
import os
import pathlib
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore", category=PossibleUserWarning)

# ---------------------------------------------------------------------------
# Path setup — script lives in scripts/, model code in DiGress/src/
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "DiGress" / "src"
sys.path.insert(0, str(SRC_DIR))

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import compose, initialize_config_dir

from diffusion_model_discrete import DiscreteDenoisingDiffusion
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from analysis.visualization import NonMolecularVisualization
from datasets.floorplan_dataset import FloorplanDataModule, FloorplanDatasetInfos
import utils


# ---------------------------------------------------------------------------
# Subset datamodule wrapper
# ---------------------------------------------------------------------------
def _make_subset(dataset, n: int, seed: int = 42) -> Subset:
    """Return a deterministic subset of exactly n samples."""
    n = min(n, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, perm[:n].tolist())


class SmallFloorplanDataModule(FloorplanDataModule):
    """Wraps FloorplanDataModule with fixed-size train and val subsets."""

    def __init__(self, cfg, n_train: int, n_val: int):
        super().__init__(cfg)
        self.n_train = n_train
        self.n_val = n_val

    def train_dataloader(self):
        subset = _make_subset(self.train_dataset, self.n_train)
        print(f"[SmallFloorplanDataModule] train: {len(subset)}/{len(self.train_dataset)} samples")
        return DataLoader(
            subset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=getattr(self.cfg.dataset, "pin_memory", False),
        )

    def val_dataloader(self):
        subset = _make_subset(self.val_dataset, self.n_val, seed=99)
        print(f"[SmallFloorplanDataModule]   val: {len(subset)}/{len(self.val_dataset)} samples")
        return DataLoader(
            subset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=getattr(self.cfg.dataset, "pin_memory", False),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train",      type=int,   default=1000)
    p.add_argument("--n_val",        type=int,   default=100)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--name",         type=str,   default="small_test")
    p.add_argument("--val_every",    type=int,   default=5)
    p.add_argument("--lr",           type=float, default=2e-4)
    return p.parse_args()


def load_cfg(args) -> DictConfig:
    config_dir = str(SRC_DIR.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config")

    # Apply overrides
    cfg.general.name = args.name
    cfg.train.n_epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.general.check_val_every_n_epochs = args.val_every
    cfg.general.sample_every_val = 999   # disable sampling during val (too slow for quick tests)
    cfg.general.wandb = "disabled"

    # Reduce model size so it trains fast on a tiny subset
    # Comment these out if you want the full-size model
    cfg.model.n_layers = 4
    cfg.model.hidden_mlp_dims = {"X": 128, "E": 64, "y": 256}
    cfg.model.hidden_dims = {"dx": 128, "de": 32, "dy": 1000, "n_head": 4,
                              "dim_ffX": 128, "dim_ffE": 32, "dim_ffy": 128}

    return cfg


def patch_dropout(model: nn.Module, p: float):
    """Insert dropout after every Linear layer inside the transformer blocks."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and p > 0:
            # We can't easily inject mid-layer, so rely on existing dropout hooks
            pass
    # Apply dropout to the full GraphTransformer by wrapping its forward
    # Simpler: just set dropout on all existing Dropout modules found
    found = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            found += 1
    if found:
        print(f"[dropout] Set p={p} on {found} existing Dropout modules.")
    else:
        print(f"[dropout] No existing Dropout modules found (p={p} has no effect). "
              f"Consider adding Dropout layers to GraphTransformer.")


def main():
    args = parse_args()

    # Change working dir to DiGress/src so Hydra/relative imports work
    os.chdir(str(SRC_DIR))

    cfg = load_cfg(args)

    print("=" * 60)
    print(f"  Small-subset training: {args.n_train} train / {args.n_val} val samples")
    print(f"  Epochs: {args.epochs} | Val every: {args.val_every}")
    print(f"  weight_decay={args.weight_decay}  dropout={args.dropout}")
    print(f"  lr={args.lr}  batch_size={args.batch_size}")
    print("=" * 60)

    datamodule = SmallFloorplanDataModule(cfg, n_train=args.n_train, n_val=args.n_val)

    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=None) \
        if cfg.model.extra_features else DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    dataset_infos = FloorplanDatasetInfos(datamodule, cfg.dataset)
    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos) \
        if cfg.model.extra_features else DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    train_metrics = TrainAbstractMetricsDiscrete()
    sampling_metrics = TrainAbstractMetricsDiscrete()
    visualization_tools = NonMolecularVisualization()

    model = DiscreteDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    # Patch weight_decay into optimizer (override configure_optimizers at runtime)
    wd = args.weight_decay

    def configure_optimizers_patched(self=model):
        return torch.optim.AdamW(
            self.parameters(), lr=cfg.train.lr, amsgrad=True, weight_decay=wd
        )

    import types
    model.configure_optimizers = types.MethodType(
        lambda self: torch.optim.AdamW(
            self.parameters(), lr=cfg.train.lr, amsgrad=True, weight_decay=wd
        ),
        model,
    )

    if args.dropout > 0:
        patch_dropout(model, args.dropout)

    # Output dir for this run
    out_dir = pathlib.Path("outputs") / "small_runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="{epoch:03d}-{val/epoch_NLL:.2f}",
        monitor="val/epoch_NLL",
        save_top_k=3,
        mode="min",
        every_n_epochs=args.val_every,
    )
    callbacks.append(checkpoint_cb)

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_every,
        gradient_clip_val=cfg.train.clip_grad,
        enable_progress_bar=True,
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=[],
        strategy="auto",
    )

    print(f"\nStarting training. Output dir: {out_dir}\n")
    trainer.fit(model, datamodule=datamodule)

    print("\n[train_small.py] Done.")
    print(f"Best val NLL: {model.best_val_nll:.4f}")
    print(f"Checkpoints saved to: {out_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
