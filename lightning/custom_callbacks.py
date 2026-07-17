import pytorch_lightning as pl


class ValidationEveryNEpochs(pl.Callback):
    def __init__(self, n: int = 10):
        super().__init__()
        self.n = max(1, int(n))

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'check_val_every_n_epoch'):
            trainer.check_val_every_n_epoch = self.n

        if hasattr(trainer, 'val_check_interval'):
            trainer.val_check_interval = None
        print(f"[Callback] 验证频率设置为每 {self.n} 个epoch 一次")

import pytorch_lightning as pl


class ValidationEveryNEpochs(pl.Callback):
    def __init__(self, n: int = 10):
        super().__init__()
        self.n = max(1, int(n))

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        if hasattr(trainer, 'check_val_every_n_epoch'):
            trainer.check_val_every_n_epoch = self.n


