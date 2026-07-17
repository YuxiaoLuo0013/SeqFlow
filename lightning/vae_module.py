import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Any, Dict

from models.vae import VAE
from utils.metrics import compute_metrics


class VAELightningModule(pl.LightningModule):
    def __init__(self,
                 num_regions: int = 2235,
                 time_steps: int = 168,
                 patch_spatial: int = 16,
                 patch_temporal: int = 24,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 tokens_per_patch: int = 7,
                 token_dim: int = 64,
                 lr: float = 2e-4,
                 weight_decay: float = 1e-4,
                 scheduler_type: str = 'cosine',
                 warmup_steps: int = 1000,
                 max_steps: int = 100000,
                 max_epochs: int = 200,
                 recon_weight: float = 1.0,
                 kl_weight: float = 0.1,
                 recon_loss: str = 'mse',
                 save_reconstructions: bool = False,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VAE(
            num_regions=num_regions,
            time_steps=time_steps,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            tokens_per_patch=tokens_per_patch,
            token_dim=token_dim
        )

        self.recon_loss_type = recon_loss
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

        from dataset.od_datamodule import LogTransformer
        self.log_transformer = LogTransformer()

        self.validation_step_outputs = []
        self._val_od_full_true = None
        self._val_od_full_pred = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self.model(x)
        return x_hat

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'global_mean') and hasattr(self, 'global_std') and self.global_mean is not None and self.global_std is not None:
            x_log = x * self.global_std + self.global_mean
            x_denorm = self.log_transformer.inverse_transform(x_log)
            return torch.clamp(x_denorm, min=0.0)
        else:
            return torch.clamp(x, min=0.0)

    def _recon_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:

            m = mask.unsqueeze(1).unsqueeze(-1).expand_as(x).float()
            diff = x_hat - x
            if self.recon_loss_type == 'mse':
                return (diff.pow(2) * m).sum() / m.sum().clamp(min=1)
            elif self.recon_loss_type == 'l1':
                return (diff.abs() * m).sum() / m.sum().clamp(min=1)
            else:
                return (F.smooth_l1_loss(x_hat, x, reduction='none') * m).sum() / m.sum().clamp(min=1)
        else:
            if self.recon_loss_type == 'mse':
                return F.mse_loss(x_hat, x)
            elif self.recon_loss_type == 'l1':
                return F.l1_loss(x_hat, x)
            else:
                return F.smooth_l1_loss(x_hat, x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        x = batch['od'] if isinstance(batch, dict) else batch
        mask = batch.get('mask') if isinstance(batch, dict) else None
        x_hat, kl_loss = self.model(x)
        recon = self._recon_loss(x, x_hat, mask=mask)
        loss = self.recon_weight * recon + self.kl_weight * kl_loss
        with torch.no_grad():
            x_den = self.denormalize(x)
            xh_den = self.denormalize(x_hat)
            if mask is not None:
                m = mask.unsqueeze(1).unsqueeze(-1).expand_as(x_den).float()
                mae = ((xh_den - x_den).abs() * m).sum() / m.sum().clamp(min=1)
                mse = ((xh_den - x_den).pow(2) * m).sum() / m.sum().clamp(min=1)
            else:
                mae = F.l1_loss(xh_den, x_den)
                mse = F.mse_loss(xh_den, x_den)
            snr = 10 * torch.log10(x.var() / (mse + 1e-8))
        self.log_dict({
            'train/loss': loss,
            'train/recon': recon,
            'train/kl': kl_loss,
            'train/mae': mae,
        }, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({
            'train/mse': mse,
            'train/snr': snr,
        }, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:

        x = batch['od'] if isinstance(batch, dict) else batch
        mask = batch.get('mask') if isinstance(batch, dict) else None
        mu, logvar = self.model.encode(x)
        x_hat = self.model.decode(mu)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl.mean()
        recon = self._recon_loss(x, x_hat, mask=mask)
        loss = self.recon_weight * recon + self.kl_weight * kl_loss
        x_den = self.denormalize(x)
        xh_den = self.denormalize(x_hat)
        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(-1).expand_as(x_den).float()
            mae = ((xh_den - x_den).abs() * m).sum() / m.sum().clamp(min=1)
            mse = ((xh_den - x_den).pow(2) * m).sum() / m.sum().clamp(min=1)
        else:
            mae = F.l1_loss(xh_den, x_den)
            mse = F.mse_loss(xh_den, x_den)
        snr = 10 * torch.log10(x.var() / (mse + 1e-8))
        self.log_dict({
            'val/loss': loss,
            'val/recon': recon,
            'val/kl': kl_loss,
            'val/mae': mae,
        }, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({
            'val/mse': mse,
            'val/snr': snr,
        }, on_step=False, on_epoch=True, prog_bar=False)

        with torch.no_grad():
            self._accumulate_metrics_from_batch(x_den.detach().cpu(), xh_den.detach().cpu(), batch_idx)
        return {'val_loss': loss}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_steps)
            return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}
        elif self.hparams.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=self.hparams.max_steps // 3, gamma=0.5)
            return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}
        else:
            return opt

    def on_validation_epoch_start(self):

        self._val_od_full_true = None
        self._val_od_full_pred = None

    def on_validation_epoch_end(self):
        import utils.metrics

        try:
            if self._val_od_full_true is None or self._val_od_full_pred is None:
                if (self.current_epoch + 1) % 10 == 0:
                    print(f"Epoch {self.current_epoch}: val CPC=0.0000, MSE=0.0000, RMSE=0.0000, NRMSE=0.0000 (no valid OD data)")
                return
            try:
                _val_od_sum_true = self._val_od_full_true.sum(axis=0)
                _val_od_sum_pred = self._val_od_full_pred.sum(axis=0)

                cpc_result = utils.metrics.compute_cpc(_val_od_sum_true, _val_od_sum_pred, mode="global", clip_negative=True)
                cpc_val = float(cpc_result.mean() if not np.isnan(cpc_result.mean()) else 0.0)

                mse_result = utils.metrics.compute_mse(self._val_od_full_true, self._val_od_full_pred, mode="global")
                rmse_result = utils.metrics.compute_rmse(self._val_od_full_true, self._val_od_full_pred, mode="global")
                nrmse_result = utils.metrics.compute_nrmse(self._val_od_full_true, self._val_od_full_pred, mode="global", normalization="rms")
                mse_val = float(mse_result.mean() if not np.isnan(mse_result.mean()) else 0.0)
                rmse_val = float(rmse_result.mean() if not np.isnan(rmse_result.mean()) else 0.0)
                nrmse_val = float(nrmse_result.mean() if not np.isnan(nrmse_result.mean()) else 0.0)

                self.log_dict({
                    'val/cpc': cpc_val,
                    'val/mse': mse_val,
                    'val/rmse': rmse_val,
                    'val/nrmse': nrmse_val,
                }, on_epoch=True, prog_bar=True)
                print(f"Epoch {self.current_epoch}: val CPC={cpc_val:.4f}, MSE={mse_val:.4f}, RMSE={rmse_val:.4f}, NRMSE={nrmse_val:.4f}")
            except Exception as e:
                print(f"Error computing validation metrics: {e}")
                self.log_dict({
                    'val/cpc': 0.0,
                    'val/mse': 0.0,
                    'val/rmse': 0.0,
                    'val/nrmse': 0.0,
                }, on_epoch=True, prog_bar=True)
        finally:
            self.validation_step_outputs.clear()
            self._val_od_full_true = None
            self._val_od_full_pred = None

    def _accumulate_metrics_from_batch(self, x_true_batch: torch.Tensor, x_pred_batch: torch.Tensor, batch_idx: int) -> None:
        """从batch中累加验证指标到完整的 (T, N, N) 矩阵"""
        try:
            if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
                return
            dm = self.trainer.datamodule
            if not hasattr(dm, 'val_set'):
                return
            val_set = dm.val_set

            N = val_set.N
            T = len(val_set.hours)
            batch_size = x_true_batch.shape[0]

            x_true_batch = x_true_batch.squeeze(1)
            x_pred_batch = x_pred_batch.squeeze(1)

            if self._val_od_full_true is None:
                self._val_od_full_true = np.zeros((T, N, N), dtype=np.float64)
                self._val_od_full_pred = np.zeros((T, N, N), dtype=np.float64)

            batch_start = batch_idx * dm.batch_size

            for i in range(batch_size):
                try:
                    val_set_idx = batch_start + i
                    if val_set_idx >= len(val_set):
                        break

                    if val_set.indices is not None:
                        actual_region_idx = val_set.indices[val_set_idx]
                    else:
                        actual_region_idx = val_set_idx

                    neighbors = val_set.neighbors[actual_region_idx]
                    origin_true = x_true_batch[i].numpy()
                    origin_pred = x_pred_batch[i].numpy()

                    actual_neighbors = len(neighbors)
                    actual_data_size = origin_true.shape[0]
                    min_size = min(actual_neighbors, actual_data_size)
                    if min_size <= 0:
                        continue

                    valid_neighbors = neighbors[:min_size]
                    valid_true = origin_true[:min_size, :]
                    valid_pred = origin_pred[:min_size, :]

                    for t in range(T):
                        self._val_od_full_true[t, actual_region_idx, valid_neighbors] += valid_true[:, t]
                        self._val_od_full_pred[t, actual_region_idx, valid_neighbors] += valid_pred[:, t]
                except Exception:
                    continue
        except Exception:
            return

    def _reconstruct_od_matrix(self, x_batch: torch.Tensor):
        try:
            if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
                return None
            dm = self.trainer.datamodule
            if not hasattr(dm, 'val_set'):
                return None
            val_set = dm.val_set
            N = val_set.N
            T = len(val_set.hours)
            N_actual = min(x_batch.shape[0], N)
            if N_actual < 1:
                return None
            x_batch = x_batch.squeeze(1)
            od_matrices = torch.zeros((T, N, N), dtype=torch.float32)
            for origin_idx in range(N_actual):
                try:
                    neighbors = val_set.neighbors[origin_idx]
                    origin_data = x_batch[origin_idx]
                    actual_neighbors = len(neighbors)
                    actual_data_size = origin_data.shape[0]
                    min_size = min(actual_neighbors, actual_data_size)
                    if min_size > 0:
                        valid_neighbors = neighbors[:min_size]
                        valid_data = origin_data[:min_size, :]
                        od_matrices[:, origin_idx, valid_neighbors] = valid_data.T
                except Exception:
                    continue
            return od_matrices.numpy()
        except Exception:
            return None


