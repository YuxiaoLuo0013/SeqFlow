import argparse
import os
import yaml
import torch
import pytorch_lightning as pl
from datetime import datetime
from pathlib import Path


torch.set_float32_matmul_precision('high')
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from lightning.vae_module import VAELightningModule
from dataset.od_datamodule import ODDataModule, MultiCityODDataModule
from lightning.custom_callbacks import ValidationEveryNEpochs


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/vae_config.yaml',
                   help='配置文件路径')
    p.add_argument('--output_dir', type=str, default=None,
                   help='训练输出目录；设置后checkpoint写入该目录下的checkpoints')
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子')
    p.add_argument('--resume_from_checkpoint', type=str, default=None,
                   help='从检查点恢复训练')
    p.add_argument('--test_only', action='store_true',
                   help='仅运行测试')
    return p.parse_args()


def main():
    args = parse_args()


    config = load_config(args.config)

    pl.seed_everything(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)


    data_config = config['data']
    if 'train_cities' in data_config:
        data = MultiCityODDataModule(
            data_dir=data_config['data_dir'],
            train_cities=data_config['train_cities'],
            test_cities=data_config.get('test_cities', []),
            norm_stats_path=data_config['norm_stats_path'],
            hours=data_config.get('hours'),
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            normalize=data_config.get('normalize', True),
            use_population=data_config.get('use_population', False),
            use_tfidf=data_config.get('use_tfidf', False),
            use_spatial_features=data_config.get('use_spatial_features', False),
            train_val_split=data_config.get('train_val_split', 0.85),
        )
        print(f"[Data] 多城市模式: 训练={data_config['train_cities']}, 测试={data_config.get('test_cities', [])}")
    else:
        data = ODDataModule(
            od_path=data_config['od_path'],
            neighbors_path=data_config['neighbors_path'],
            hours=data_config.get('hours', None),
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            normalize=data_config.get('normalize', True),
            clip_max=data_config.get('clip_max', None),
        )
        print("[Data] 单城市模式")


    model_config = config['model']
    training_config = config['training']

    model = VAELightningModule(
        num_regions=model_config['num_regions'],
        time_steps=model_config['time_steps'],
        patch_spatial=model_config['patch_spatial'],
        patch_temporal=model_config['patch_temporal'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config['mlp_ratio'],
        dropout=model_config['dropout'],
        tokens_per_patch=model_config['tokens_per_patch'],
        token_dim=model_config['token_dim'],
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        scheduler_type=training_config['scheduler_type'],
        warmup_steps=training_config['warmup_steps'],
        max_steps=training_config['max_steps'],
        max_epochs=training_config.get('max_epochs', 200),
        recon_weight=training_config['recon_weight'],
        kl_weight=training_config.get('kl_weight', 0.1),
        recon_loss=training_config.get('recon_loss', 'mse'),
        save_reconstructions=training_config.get('save_reconstructions', False),
    )


    logging_config = config['logging']
    logger_save_dir = str(output_dir / "logs") if output_dir is not None else logging_config['save_dir']
    logger = TensorBoardLogger(
        save_dir=logger_save_dir,
        name=logging_config['name'],
        version=logging_config.get('version', None)
    )


    callbacks = []


    callbacks.append(ValidationEveryNEpochs(n=config.get('trainer', {}).get('check_val_every_n_epoch', 10)))


    checkpoint_callback = None
    if 'callbacks' in config and 'model_checkpoint' in config['callbacks']:
        ckpt_config = config['callbacks']['model_checkpoint']
        ckpt_dir = output_dir / "checkpoints" if output_dir is not None else Path('checkpoints')
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        time_str = datetime.now().strftime('%Y%m%d-%H%M%S')

        filename = ckpt_config.get('filename', f"vae-{time_str}-{{epoch:02d}}")
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor=ckpt_config['monitor'],
            mode=ckpt_config['mode'],
            save_top_k=ckpt_config.get('save_top_k', 1),
            save_last=ckpt_config.get('save_last', False),
            filename=filename,
            auto_insert_metric_name=ckpt_config.get('auto_insert_metric_name', False),
            save_on_train_epoch_end=ckpt_config.get('save_on_train_epoch_end', None),
        )
        callbacks.append(checkpoint_callback)


    if 'callbacks' in config and 'early_stopping' in config['callbacks']:
        es_config = config['callbacks']['early_stopping']
        early_stopping = EarlyStopping(
            monitor=es_config['monitor'],
            mode=es_config['mode'],
            patience=es_config['patience'],
            min_delta=es_config['min_delta'],
            check_on_train_epoch_end=es_config.get('check_on_train_epoch_end', False),
        )
        callbacks.append(early_stopping)


    if 'callbacks' in config and 'learning_rate_monitor' in config['callbacks']:
        lr_config = config['callbacks']['learning_rate_monitor']
        lr_monitor = LearningRateMonitor(
            logging_interval=lr_config['logging_interval']
        )
        callbacks.append(lr_monitor)


    trainer_config = config['trainer']
    precision = trainer_config.get('precision', 32)
    if precision == 16:
        precision = '16-mixed'
    elif precision == 'bf16':
        precision = 'bf16-mixed'
    else:
        precision = '32-true'

    trainer = pl.Trainer(
        max_epochs=training_config.get('max_epochs', 100),
        max_steps=training_config.get('max_steps', -1),
        devices=trainer_config['devices'],
        accelerator=trainer_config['accelerator'],
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=trainer_config.get('gradient_clip_val', None),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        check_val_every_n_epoch=trainer_config.get('check_val_every_n_epoch', 1),
        enable_checkpointing=trainer_config.get('enable_checkpointing', True),
        enable_progress_bar=trainer_config.get('enable_progress_bar', True),
        enable_model_summary=trainer_config.get('enable_model_summary', True),
        deterministic=trainer_config.get('deterministic', False),
        benchmark=trainer_config.get('benchmark', True),
        log_every_n_steps=training_config.get('log_every_n_steps', 50),
        val_check_interval=training_config.get('val_check_interval', 1.0),
        limit_val_batches=trainer_config.get('limit_val_batches', 1.0),
        num_sanity_val_steps=trainer_config.get('num_sanity_val_steps', 2),
    )



    data.setup()

    if data.train_set.normalize:
        if hasattr(data.train_set, 'global_mean') and hasattr(data.train_set, 'global_std') and \
           data.train_set.global_mean is not None and data.train_set.global_std is not None:

            model.global_min = None
            model.global_max = None
            model.global_mean = float(data.train_set.global_mean)
            model.global_std = float(data.train_set.global_std)
            print(f"已设置模型归一化统计信息: 均值={model.global_mean:.6f}, 标准差={model.global_std:.6f}")
        elif hasattr(data.train_set, 'global_min') and hasattr(data.train_set, 'global_max'):
            model.set_normalization_stats(data.train_set.global_min, data.train_set.global_max)
            print(f"已设置模型归一化统计信息(回退MinMax): 最小值={data.train_set.global_min:.6f}, 最大值={data.train_set.global_max:.6f}")





    trainer.fit(model, datamodule=data, ckpt_path=args.resume_from_checkpoint)

    if output_dir is not None and checkpoint_callback is not None:
        best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        if not best_path:
            raise RuntimeError("No VAE checkpoint was saved.")
        with open(output_dir / "best_checkpoint.txt", "w", encoding="utf-8") as f:
            f.write(best_path + "\n")
        print(f"[train_vae] best checkpoint: {best_path}")


if __name__ == '__main__':
    main()
