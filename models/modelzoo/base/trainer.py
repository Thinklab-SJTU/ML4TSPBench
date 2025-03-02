import os
import numpy as np
import torch
from typing import Union, Optional
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning_utilities.core.rank_zero import rank_zero_info
from .nar import TSPNAREncoder
from .logger import Logger
from .checkpoint import Checkpoint
from models.rl4co.common.models_base import TSPAREncoder


class TSPNARTrainer(Trainer):
    def __init__(
        self,
        model: TSPNAREncoder,
        logger: Optional[Logger]=None,
        wandb_logger_name: str="wandb",
        resume_id: Optional[str]=None,
        ckpt_save_path: Optional[str]=None,
        monitor: str="val/loss",
        every_n_epochs: int=1,
        every_n_train_steps: Optional[int]=None,
        val_check_interval: Optional[int]=None,
        log_every_n_steps: Optional[int]=50,
        accelerator: str="auto",
        strategy: Union[str, Strategy]=DDPStrategy(static_graph=True),
        max_epochs: int=100,
        max_steps: int=-1,
        fp16: bool=False,
        ckpt_path: Optional[str]=None,
        **kwargs
    ):
        if logger is None:
            self.logger = Logger(name=wandb_logger_name, resume_id=resume_id)
        else:
            self.logger = logger
        if ckpt_save_path is None:
            self.ckpt_save_path = os.path.join("train_ckpts", self.logger._name, self.logger._id)
        self.ckpt_callback = Checkpoint(
            dirpath=self.ckpt_save_path,
            monitor=monitor,
            every_n_epochs=every_n_epochs if kwargs['encoder'] != 'dimes' else 1,
            every_n_train_steps=every_n_train_steps
        )
        self.lr_callback = LearningRateMonitor(logging_interval='step')
        
        devices=torch.cuda.device_count() if torch.cuda.is_available() else "auto"
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=16 if fp16 else 32,
            logger=self.logger,
            callbacks=[TQDMProgressBar(refresh_rate=20), self.ckpt_callback, self.lr_callback],
            max_epochs=max_epochs,
            max_steps=max_steps,
            check_val_every_n_epoch=1,
            val_check_interval=val_check_interval,
            log_every_n_steps=log_every_n_steps,
            inference_mode=False if kwargs['encoder'] == 'dimes' or \
                                    kwargs['active_search'] else True
        )
        
        if ckpt_path is not None:
            model.load_ckpt(ckpt_path)

        self.train_model = model
    
    def model_train(self):
        rank_zero_info(f"Logging to {self.logger.save_dir}/{self.logger.name}/{self.logger.version}")
        rank_zero_info(f"checkpoint_callback's dirpath is {self.ckpt_save_path}")
        rank_zero_info(
            f"{'-' * 100}\n"
            f"{str(self.train_model)}\n"
            f"{'-' * 100}\n"
        )
        self.fit(self.train_model)
        self.logger.finalize("success")
    
    def model_test(self):
        rank_zero_info(f"Logging to {self.logger.save_dir}/{self.logger.name}/{self.logger.version}")
        rank_zero_info(
            f"{'-' * 100}\n"
            f"{str(self.train_model)}\n"
            f"{'-' * 100}\n"
        )
        self.test(self.train_model)
        print("std: ", torch.std(torch.tensor(self.train_model.gap_list)))
        
        # avg_decoding_time = self.train_model.decoding_time / len(self.test_dataloaders)
        # with open('tmp1.txt', 'a') as file1:
        #     file1.write(str(avg_decoding_time) + ' ')
            
        # avg_gap = np.average(self.train_model.gap_list)
        # with open('tmp2.txt', 'a') as file2:
        #     file2.write(str(avg_gap) + ' ')
        

class TSPARTrainer(Trainer):
    def __init__(
        self,
        model: TSPAREncoder,
        logger: Logger=None,
        wandb_logger_name: str="wandb",
        resume_id: str=None,
        ckpt_save_path: str=None,
        monitor: str="val/loss",
        every_n_epochs: int=1,
        accelerator: str="auto",
        strategy: Union[str, Strategy]="auto",
        max_epochs: int=100,
        gradient_clip_val: Union[int, float] = 1.0,
        fp16: bool=False,
        ckpt_path: str=None,
        **kwargs
    ):
        if logger is None:
            self.logger = Logger(name=wandb_logger_name, resume_id=resume_id)
        else:
            self.logger = logger
        if ckpt_save_path is None:
            self.ckpt_save_path = os.path.join("train_ckpts", self.logger._name, self.logger._id)
        else:
            self.ckpt_save_path = os.path.join("train_ckpts", ckpt_save_path)
        self.ckpt_callback = Checkpoint(
            dirpath=self.ckpt_save_path,
            monitor=monitor,
            every_n_epochs=every_n_epochs
        )
        self.lr_callback = LearningRateMonitor(logging_interval='step')
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
            precision=16 if fp16 else 32,
            logger=self.logger,
            callbacks=[TQDMProgressBar(refresh_rate=20), self.ckpt_callback, self.lr_callback],
            max_epochs=max_epochs,
            check_val_every_n_epoch=1,
            gradient_clip_val=gradient_clip_val,
            reload_dataloaders_every_n_epochs=1
        )
        
        if ckpt_path is not None:
            try:
                model.load_from_checkpoint(ckpt_path)
            except:
                model.load_ckpt(ckpt_path)
        self.train_model = model
    
    def model_train(self):
        rank_zero_info(f"Logging to {self.logger.save_dir}/{self.logger.name}/{self.logger.version}")
        rank_zero_info(f"checkpoint_callback's dirpath is {self.ckpt_save_path}")
        rank_zero_info(
            f"{'-' * 100}\n"
            f"{str(self.train_model)}\n"
            f"{'-' * 100}\n"
        )
        self.fit(self.train_model)
        self.logger.finalize("success")
    
    def model_test(self):
        rank_zero_info(f"Logging to {self.logger.save_dir}/{self.logger.name}/{self.logger.version}")
        rank_zero_info(
            f"{'-' * 100}\n"
            f"{str(self.train_model)}\n"
            f"{'-' * 100}\n"
        )
        self.test(self.train_model)
        print("std: ", torch.std(torch.tensor(self.train_model.gap_list)))