from pytorch_lightning.callbacks import ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath: str="wandb/checkpoints",
        monitor: str="val/loss",
        every_n_epochs: int=1,
        every_n_train_steps=None,
        filename='epoch={epoch}-step={step}-gap={val/gap:.5f}'
    ):
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            mode="min",
            save_top_k=-1,
            save_last=True,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=every_n_train_steps,
            filename=filename,
            auto_insert_metric_name=False
        )
