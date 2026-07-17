
import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.od_datamodule import MultiCityODDataModule
from lightning.flow_ar_module import FlowARLightningModule
from utils import metrics as od_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_module(config: dict) -> FlowARLightningModule:
    vae_cfg = config["vae"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    return FlowARLightningModule(
        num_regions=vae_cfg["num_regions"],
        time_steps=vae_cfg["time_steps"],
        patch_spatial=vae_cfg["patch_spatial"],
        patch_temporal=vae_cfg["patch_temporal"],
        embed_dim=vae_cfg["embed_dim"],
        num_encoder_layers=vae_cfg["num_encoder_layers"],
        num_decoder_layers=vae_cfg["num_decoder_layers"],
        num_heads=vae_cfg["num_heads"],
        mlp_ratio=vae_cfg["mlp_ratio"],
        dropout=vae_cfg["dropout"],
        tokens_per_patch=vae_cfg["tokens_per_patch"],
        token_dim=vae_cfg["token_dim"],
        model_dim=model_cfg["model_dim"],
        depth=model_cfg["depth"],
        ar_heads=model_cfg["ar_heads"],
        ar_mlp_ratio=model_cfg["ar_mlp_ratio"],
        attn_drop=model_cfg["attn_drop"],
        proj_drop=model_cfg["proj_drop"],
        cond_channels=model_cfg.get("cond_channels", 266),
        spatial_dim=model_cfg.get("spatial_dim", 5),
        max_group_len=model_cfg["max_group_len"],
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
        scheduler_type=training_cfg["scheduler_type"],
        warmup_steps=training_cfg["warmup_steps"],
        max_steps=training_cfg["max_steps"],
        loss_rel_weight=training_cfg.get("loss_rel_weight", 0.01),
        loss_struct_weight=training_cfg.get("loss_struct_weight", 0.01),
        loss_aux_warmup_epochs=training_cfg.get("loss_aux_warmup_epochs", 30),
        vae_ckpt=vae_cfg["ckpt"],
    )


def build_datamodule(config: dict) -> MultiCityODDataModule:
    data_cfg = config["data"]
    return MultiCityODDataModule(
        data_dir=data_cfg["data_dir"],
        train_cities=data_cfg["train_cities"],
        test_cities=data_cfg.get("test_cities", []),
        norm_stats_path=data_cfg["norm_stats_path"],
        hours=data_cfg.get("hours"),
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        normalize=data_cfg.get("normalize", True),
        use_population=data_cfg.get("use_population", True),
        use_tfidf=data_cfg.get("use_tfidf", False),
        use_spatial_features=data_cfg.get("use_spatial_features", True),
        train_val_split=data_cfg.get("train_val_split", 0.8),
    )


def set_norm_stats(module: FlowARLightningModule, dm: MultiCityODDataModule) -> None:
    module.global_mean = float(dm.train_set.global_mean)
    module.global_std = float(dm.train_set.global_std)


def trainer_precision(config: dict) -> str:
    precision = config["training"].get("precision", "32-true")
    if precision in [16, "16", "16-mixed"]:
        return "16-mixed"
    if precision in ["bf16", "bf16-mixed"]:
        return "bf16-mixed"
    if precision in [32, "32", "32-true"]:
        return "32-true"
    return precision


class MinEpochEarlyStopping(EarlyStopping):
    def __init__(self, *args, min_monitor_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_monitor_epoch = int(min_monitor_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 < self.min_monitor_epoch:
            return
        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 < self.min_monitor_epoch:
            return
        return super().on_validation_end(trainer, pl_module)


class MinEpochModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, min_checkpoint_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_checkpoint_epoch = int(min_checkpoint_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 < self.min_checkpoint_epoch:
            return
        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 < self.min_checkpoint_epoch:
            return
        return super().on_validation_end(trainer, pl_module)


def build_callbacks(config: dict, output_dir: Path):
    cb_cfg = config.get("callbacks", {})
    callbacks = []
    ckpt_cb = None
    if "model_checkpoint" in cb_cfg:
        cfg = cb_cfg["model_checkpoint"]
        ckpt_cb = MinEpochModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            monitor=cfg.get("monitor", "val/cpc"),
            mode=cfg.get("mode", "max"),
            save_top_k=cfg.get("save_top_k", 1),
            save_last=cfg.get("save_last", True),
            filename="fold-{epoch:04d}",
            auto_insert_metric_name=cfg.get("auto_insert_metric_name", False),
            save_on_train_epoch_end=cfg.get("save_on_train_epoch_end", None),
            min_checkpoint_epoch=int(config["training"].get("min_checkpoint_epoch", 0)),
        )
        callbacks.append(ckpt_cb)
    if "early_stopping" in cb_cfg:
        cfg = cb_cfg["early_stopping"]
        callbacks.append(
            MinEpochEarlyStopping(
                monitor=cfg.get("monitor", "val/cpc"),
                mode=cfg.get("mode", "max"),
                patience=int(cfg.get("patience", 20)),
                min_delta=float(cfg.get("min_delta", 0.0)),
                check_on_train_epoch_end=cfg.get("check_on_train_epoch_end", False),
                min_monitor_epoch=int(cfg.get("min_monitor_epoch", 0)),
            )
        )
    if "learning_rate_monitor" in cb_cfg:
        callbacks.append(
            LearningRateMonitor(logging_interval=cb_cfg["learning_rate_monitor"].get("logging_interval", "step"))
        )
    return callbacks, ckpt_cb


def sample_batch(module: FlowARLightningModule, batch: Dict[str, Any], device: torch.device):
    x_od = batch["od"].to(device)
    condition = batch["condition"]
    condition = condition.to(device) if condition is not None else None
    groups = module._build_group_lengths()
    with torch.no_grad():
        if condition is None:
            condition = torch.zeros(
                x_od.shape[0], module.hparams.cond_channels, module.hparams.num_regions + 1, 1, device=device
            )
        sampled_tokens = module.model.sample(
            pop_condition=condition,
            groups=groups,
        )
        sampled_tokens = sampled_tokens.reshape(
            sampled_tokens.shape[0], -1, module.hparams.tokens_per_patch, module.hparams.token_dim
        )
        pred = module.vae.decode(sampled_tokens)
        true_den = module.denormalize(x_od)
        pred_den = module.denormalize(pred)
    return true_den.detach().cpu(), pred_den.detach().cpu()


def generate_and_score(config: dict, dm: MultiCityODDataModule, ckpt_path: str, output_dir: Path) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = FlowARLightningModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    set_norm_stats(module, dm)
    module.to(device)
    module.eval()

    test_set = dm.test_set
    if len(test_set.cities) != 1:
        raise ValueError("This fold runner expects exactly one test city.")
    city = test_set.cities[0]
    city_name = city["name"]
    n = city["N"]
    t = len(test_set.hours)
    true_full = np.zeros((t, n, n), dtype=np.float32)
    pred_full = np.zeros((t, n, n), dtype=np.float32)

    loader = dm.test_dataloader()
    batch_size = dm.batch_size
    limit_test_batches = int(config.get("evaluation", {}).get("limit_test_batches", 0) or 0)
    for batch_idx, batch in enumerate(loader):
        if limit_test_batches > 0 and batch_idx >= limit_test_batches:
            break
        true_batch, pred_batch = sample_batch(module, batch, device)
        true_batch = true_batch.squeeze(1).numpy()
        pred_batch = np.maximum(pred_batch.squeeze(1).numpy(), 0.0)
        start = batch_idx * batch_size
        for i in range(true_batch.shape[0]):
            local_idx = start + i
            if local_idx >= n:
                break
            origin = int(city["tile_ids"][local_idx])
            neighbors = city["neighbors"][local_idx]
            valid = min(n - 1, true_batch.shape[1], len(neighbors))
            dest = neighbors[:valid]
            true_full[:, origin, dest] = true_batch[i, :valid, :].T
            pred_full[:, origin, dest] = pred_batch[i, :valid, :].T
        print(f"[test] {city_name} batch {batch_idx + 1}/{len(loader)}", flush=True)

    matrix_path = output_dir / "od_matrices.npz"
    np.savez_compressed(
        matrix_path,
        generated_od=pred_full,
        ground_truth_od=true_full,
        city=np.array(city_name),
        hours=np.asarray(test_set.hours, dtype=np.int32),
    )

    metric_values = od_metrics.compute_metrics(
        true_full,
        pred_full,
        mode="global",
        nrmse_norm="rms",
        clip_negative=True,
    )
    cpc = float(metric_values["cpc"].mean())
    mse = float(metric_values["mse"].mean())
    rmse = float(metric_values["rmse"].mean())
    nrmse = float(metric_values["nrmse"].mean())
    jsd_inflow = float(metric_values["jsd_inflow"].mean())
    jsd_outflow = float(metric_values["jsd_outflow"].mean())
    jsd_odflow = float(metric_values["jsd_odflow"].mean())
    mae = float(np.mean(np.abs(true_full - pred_full)))
    t_true = true_full.reshape(t, -1).sum(axis=1)
    t_pred = pred_full.reshape(t, -1).sum(axis=1)
    t_mse = float(np.abs(t_true - t_pred).mean())

    result = {
        "city": city_name,
        "checkpoint": ckpt_path,
        "matrix_path": str(matrix_path),
        "cpc": cpc,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "jsd_inflow": jsd_inflow,
        "jsd_outflow": jsd_outflow,
        "jsd_odflow": jsd_odflow,
        "t_mse": t_mse,
        "generated_sum": float(pred_full.sum()),
        "ground_truth_sum": float(true_full.sum()),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    print("[test] saved", matrix_path, flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = build_datamodule(config)
    dm.setup()
    module = build_module(config)
    set_norm_stats(module, dm)

    callbacks, checkpoint_callback = build_callbacks(config, output_dir)
    logging_cfg = config.get("logging", {})
    logger = TensorBoardLogger(
        save_dir=logging_cfg.get("save_dir", str(output_dir / "logs")),
        name=logging_cfg.get("name", output_dir.name),
        version=logging_cfg.get("version"),
    )
    trainer_cfg = config.get("trainer", {})
    training_cfg = config["training"]
    trainer = pl.Trainer(
        max_epochs=int(training_cfg.get("epochs", 1000)),
        min_epochs=int(training_cfg.get("min_epochs", 0)),
        max_steps=int(training_cfg.get("max_steps", -1)),
        devices=trainer_cfg.get("devices", [0]),
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        precision=trainer_precision(config),
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 3),
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        enable_checkpointing=trainer_cfg.get("enable_checkpointing", True),
        enable_progress_bar=trainer_cfg.get("enable_progress_bar", False),
        enable_model_summary=trainer_cfg.get("enable_model_summary", True),
        deterministic=trainer_cfg.get("deterministic", False),
        benchmark=trainer_cfg.get("benchmark", True),
        log_every_n_steps=training_cfg.get("log_every_n_steps", 50),
        num_sanity_val_steps=trainer_cfg.get("num_sanity_val_steps", 0),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        default_root_dir=str(output_dir),
    )
    print(f"[train] start {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"[train] train={config['data']['train_cities']} test={config['data']['test_cities']}", flush=True)
    trainer.fit(module, datamodule=dm)

    best_path = ""
    if checkpoint_callback is not None:
        best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    if not best_path:
        raise RuntimeError("No checkpoint was saved.")
    with open(output_dir / "best_checkpoint.txt", "w", encoding="utf-8") as f:
        f.write(best_path + "\n")
    print(f"[train] best checkpoint: {best_path}", flush=True)
    generate_and_score(config, dm, best_path, output_dir)

    if bool(training_cfg.get("delete_checkpoints_after_eval", False)):
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            print(f"[cleanup] removed {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
