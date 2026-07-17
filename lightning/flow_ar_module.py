import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Any, Dict, List, Optional

from models.vae import VAE
from models.flow_ar_distance import FlowARDistance
from utils.patch_scheduler import group_lengths
from dataset.od_datamodule import LogTransformer


class FlowARLightningModule(pl.LightningModule):
    def __init__(
        self,

        num_regions: int = 2235,
        time_steps: int = 168,
        patch_spatial: int = 16,
        patch_temporal: int = 24,
        embed_dim: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        tokens_per_patch: int = 7,
        token_dim: int = 64,

        model_dim: int = 768,
        depth: int = 6,
        ar_heads: int = 12,
        ar_mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        cond_channels: int = 266,
        spatial_dim: int = 5,

        max_group_len: int = 256,

        vae_ckpt: Optional[str] = None,

        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        warmup_steps: int = 1000,
        max_steps: int = -1,
        loss_rel_weight: float = 0.01,
        loss_struct_weight: float = 0.01,
        loss_aux_warmup_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()


        self.vae = VAE(
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
            token_dim=token_dim,
        )
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()


        if vae_ckpt is not None and len(str(vae_ckpt)) > 0:
            self._load_vae_checkpoint(vae_ckpt)


        self.model = FlowARDistance(
            token_dim=token_dim,
            model_dim=model_dim,
            depth=depth,
            num_heads=ar_heads,
            mlp_ratio=ar_mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            num_tokens_per_patch=tokens_per_patch,
            patch_spatial=patch_spatial,
            cond_channels=cond_channels,
            spatial_channels=spatial_dim,
        )


        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None
        self.log_transformer = LogTransformer()


        self._val_od_full_true = None
        self._val_od_full_pred = None


        self.loss_aux_warmup_epochs = int(loss_aux_warmup_epochs)
        self.max_loss_rel_weight = float(loss_rel_weight)
        self.max_loss_struct_weight = float(loss_struct_weight)

    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """将输入张量经VAE编码为潜变量tokens（使用`mu`作为embedding）。"""
        with torch.no_grad():
            tokens = self.vae.encoder(x)
            mu = self.vae.to_mu(tokens)
        return mu

    def get_aux_loss_weight(self, max_weight: float) -> float:
        """
        计算当前epoch的辅助loss权重
        - 从训练一开始线性增加
        - 在 self.loss_aux_warmup_epochs 时达到 max_weight，之后保持不变
        """
        warmup_epochs = max(1, self.loss_aux_warmup_epochs)
        progress = max(0.0, min(1.0, self.current_epoch / warmup_epochs))
        return progress * float(max_weight)

    def get_loss_rel_weight(self) -> float:
        return self.get_aux_loss_weight(self.max_loss_rel_weight)

    def get_loss_struct_weight(self) -> float:
        return self.get_aux_loss_weight(self.max_loss_struct_weight)

    def _build_group_lengths(self) -> List[int]:
        """
        根据VAE配置计算正确的分组长度：
        - 每个group的token数量 = tokens_per_patch
        - group数量 = ceil(num_regions / patch_spatial)

        注意：此方法不需要gt数据，仅基于配置参数计算，确保验证时不会泄露gt信息。
        """
        tokens_per_patch = self.hparams.tokens_per_patch
        num_regions = self.hparams.num_regions
        patch_spatial = self.hparams.patch_spatial


        num_spatial_patches = (num_regions + patch_spatial - 1) // patch_spatial


        groups = [tokens_per_patch] * num_spatial_patches
        return groups

    def _load_vae_checkpoint(self, ckpt_path: str) -> None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            sub = {}
            prefix = "model."
            for k, v in state.items():
                if k.startswith(prefix):
                    sub[k[len(prefix) :]] = v


            missing, unexpected = self.vae.load_state_dict(sub, strict=False)

            if len(missing) > 0 or len(unexpected) > 0:

                print("[VAE CKPT] missing keys:", missing)
                print("[VAE CKPT] unexpected keys:", unexpected)
                raise RuntimeError(
                    f"[VAE CKPT] state_dict mismatch: {len(missing)} missing, "
                    f"{len(unexpected)} unexpected. 请检查 VAE 结构配置（如 token_dim / "
                    "tokens_per_patch / 层数等）是否与训练该 ckpt 时完全一致。"
                )

            print("[VAE CKPT] loaded from", ckpt_path)
        except Exception as e:


            print("[VAE CKPT] load failed:", e)
            raise

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        if isinstance(batch, dict):
            x_od = batch['od']
            condition_vec = batch['condition']
            groups = None
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, groups = batch
            x_od = x
            condition_vec = None
        else:
            x_od = batch
            condition_vec = None
            groups = None

        mu = self.encode_to_tokens(x_od)
        lengths = group_lengths(groups) if groups is not None else self._build_group_lengths()
        token_loss,x0_pred = self.model(gt_tokens=mu, groups=lengths, pop_condition=condition_vec)
        x0_pred = x0_pred.reshape(x0_pred.shape[0], -1, self.hparams.tokens_per_patch, self.hparams.token_dim)
        x_hat = self.vae.decode(x0_pred)
        x_od = self.denormalize(x_od)
        x_hat = self.denormalize(x_hat)


        eps = 1e-8


        x_hat_pos = torch.clamp(x_hat, min=0.0)
        x_od_pos  = torch.clamp(x_od,  min=0.0)



        V_hat  = x_hat_pos.sum(dim=2)
        V_true = x_od_pos.sum(dim=2)

        tau = 0.01 * V_true.abs().mean().detach()
        denom = V_true.abs().clamp(min=tau)

        mask_rel = (V_true.abs() > tau)
        rel_err = (V_hat - V_true).abs() / denom
        loss_rel = rel_err.mean()

        w_rel = self.get_loss_rel_weight()



        col_sum_hat  = x_hat_pos.sum(dim=2, keepdim=True)
        col_sum_true = x_od_pos.sum(dim=2, keepdim=True)

        S_hat  = x_hat_pos / (col_sum_hat + eps)
        S_true = x_od_pos  / (col_sum_true + eps)


        struct_err = (S_hat - S_true).abs().sum(dim=2)

        tau_struct = 0.01 * col_sum_true.mean().detach()
        mask_struct = (col_sum_true.squeeze(2) > tau_struct)
        loss_struct =  struct_err.mean()

        w_struct = self.get_loss_struct_weight()


        loss = token_loss+w_rel*loss_rel+w_struct*loss_struct


        self.log('train/token_loss', token_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_rel', loss_rel, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_rel_weight', w_rel, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_struct', loss_struct, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

















































    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:

        if isinstance(batch, dict):
            x_od = batch['od']
            condition_vec = batch['condition']
            groups = None
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, groups = batch
            x_od = x
            condition_vec = None
        else:
            x_od = batch
            condition_vec = None
            groups = None


        lengths = group_lengths(groups) if groups is not None else self._build_group_lengths()


        with torch.no_grad():
            if condition_vec is not None:

                sampled_tokens = self.model.sample(
                    pop_condition=condition_vec,
                    groups=lengths,
                )
            else:

                B = x_od.shape[0]
                dummy_cond = torch.zeros(B, self.hparams.cond_channels, 2236, 1, device=x_od.device)
                sampled_tokens = self.model.sample(
                    pop_condition=dummy_cond,
                    groups=lengths,
                )


            sampled_tokens = sampled_tokens.reshape(sampled_tokens.shape[0], -1, self.hparams.tokens_per_patch, self.hparams.token_dim)
            x_hat = self.vae.decode(sampled_tokens)

            x_den = self.denormalize(x_od)
            xh_den = self.denormalize(x_hat)


            mae = F.l1_loss(xh_den, x_den)
            mse = F.mse_loss(xh_den, x_den)

            self.log_dict({
                'val/mae': mae,
                'val/mse': mse
            }, on_step=False, on_epoch=True, prog_bar=True)


            self._accumulate_metrics_from_batch(x_den.detach().cpu(), xh_den.detach().cpu(), batch_idx)

        return {
            'val_mae': mae.detach(),
            'val_mse': mse.detach()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        if self.hparams.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_steps if self.hparams.max_steps > 0 else 1000
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        elif self.hparams.scheduler_type == "warmup_cosine":
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(step):
                if step < self.hparams.warmup_steps:
                    return step / self.hparams.warmup_steps
                else:
                    total_steps = self.hparams.max_steps if self.hparams.max_steps > 0 else 1000
                    progress = (step - self.hparams.warmup_steps) / (total_steps - self.hparams.warmup_steps)
                    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

            scheduler = LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def on_fit_start(self) -> None:

        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, 'train_set') and dm.train_set is not None:
                if getattr(dm.train_set, 'normalize', False):
                    self.global_mean = dm.train_set.global_mean
                    self.global_std = dm.train_set.global_std

    def on_validation_epoch_start(self):

        self._val_od_full_true = None
        self._val_od_full_pred = None

    def on_validation_epoch_end(self):
        import numpy as np
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

                _val_od_sum_true_t=self._val_od_full_true.reshape(self._val_od_full_true.shape[0], -1).sum(axis=1)
                _val_od_sum_pred_t=self._val_od_full_pred.reshape(self._val_od_full_pred.shape[0], -1).sum(axis=1)
                t_mse=np.abs(_val_od_sum_true_t - _val_od_sum_pred_t).mean()
                self.log_dict({
                    'val/cpc': cpc_val,
                    'val/mse': mse_val,
                    'val/rmse': rmse_val,
                    'val/nrmse': nrmse_val,
                    'val/t_mse': t_mse,
                }, on_epoch=True, prog_bar=True)

                print(f"Epoch {self.current_epoch}: val CPC={cpc_val:.4f}, MSE={mse_val:.4f}, RMSE={rmse_val:.4f}, NRMSE={nrmse_val:.4f}, t_mse={t_mse:.4f}")
            except Exception as e:
                print(f"Error computing validation metrics: {e}")
                self.log_dict({
                    'val/cpc': 0.0,
                    'val/mse': 0.0,
                    'val/rmse': 0.0,
                    'val/nrmse': 0.0,
                    'val/t_mse': 0.0,
                }, on_epoch=True, prog_bar=True)
        finally:
            self._val_od_full_true = None
            self._val_od_full_pred = None

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:

        if self.global_mean is not None and self.global_std is not None:
            x_log = x * self.global_std + self.global_mean
            x_denorm = self.log_transformer.inverse_transform(x_log)
            return torch.clamp(x_denorm, min=0.0)
        return torch.clamp(x, min=0.0)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):

        if isinstance(batch, dict):
            x_od = batch['od']
            condition_vec = batch['condition']
        elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x_od = batch[0]
            condition_vec = None
        else:
            x_od = batch
            condition_vec = None


        groups = self._build_group_lengths()


        if condition_vec is not None:

            sampled_tokens = self.model.sample(
                pop_condition=condition_vec,
                groups=groups,
            )
        else:

            B = x_od.shape[0]
            dummy_cond = torch.zeros(B, self.hparams.cond_channels, 2236, 1, device=x_od.device)
            sampled_tokens = self.model.sample(
                pop_condition=dummy_cond,
                groups=groups,
            )


        with torch.no_grad():
            x_hat = self.vae.decode(sampled_tokens)


        x_den = self.denormalize(x_od)
        xh_den = self.denormalize(x_hat)
        mae = F.l1_loss(xh_den, x_den)
        mse = F.mse_loss(xh_den, x_den)
        self.log_dict({'pred/mae': mae, 'pred/mse': mse}, on_step=False, on_epoch=True, prog_bar=True)
        return {'mae': mae.detach(), 'mse': mse.detach()}

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
