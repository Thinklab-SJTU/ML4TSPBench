import os
from wandb.util import generate_id
from pytorch_lightning.loggers.wandb import WandbLogger
from typing import Optional


class Logger(WandbLogger):
    def __init__(
        self,
        name: str="wandb",
        project: str="ml4tspbench",
        entity: Optional[str]=None,
        save_dir: str="log",
        id: Optional[str]=None,
        resume_id: Optional[str]=None
    ):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if id is None and resume_id is None:
            wandb_id = os.getenv("WANDB_RUN_ID") or generate_id()
        else:
            wandb_id = id if id is not None else resume_id
        super().__init__(
            name=name,
            project=project,
            entity=entity,
            save_dir=save_dir,
            id=wandb_id
        )
