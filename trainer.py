import lightning.pytorch as pl
import torch
import torch.nn as nn

from model import initialize_weights
from torchmetrics import AUROC

from pathlib import Path 
import pandas as pd
import numpy as np

from training_config import TrainingConfig
from utils import CODES, get_weigths_from_pwm

class LitModel(pl.LightningModule):
    def __init__(self, tr_cfg: TrainingConfig):
        super().__init__()
        
        self.tr_cfg = tr_cfg
        self.max_lr=self.tr_cfg.max_lr
        self.model = self.tr_cfg.get_model()
        self.loss = nn.BCEWithLogitsLoss() 
        self.metric = AUROC(task="binary")
        self.metric_name = 'auroc'
        self.sigmoid = nn.Sigmoid()
    
    def set_stem_requires_grad(self, requires_grad=None):
        if requires_grad is None:
            requires_grad = self.tr_cfg.pwms_freeze
        print('Unfreezing' if requires_grad else 'Freezing', 'stem conv layer')
        for param in self.model.pwmlike_layer.parameters():
            param.requires_grad = requires_grad
    
    def initialize_weights(self):
        self.model.apply(initialize_weights)
        self.initialize_stem_with_pwms()
        
    def initialize_stem_with_pwms(self):
        if self.tr_cfg.pwms_path is None:
            return
        print('Initializing stem conv layer with PWMs...')
        pwms_path = Path(self.tr_cfg.pwms_path)
        pwm_paths = list(pwms_path.rglob('*/*.pwm'))
        pwm_count = len(pwm_paths)
        if pwm_count == 0:
            raise Exception('No PWMs were found')
        print(f'Initializing {pwm_count} PWMs')
        if self.tr_cfg.stem_ch < pwm_count:
            pwm_paths = pwm_paths[:self.tr_cfg.stem_ch//2]
        pwmlike_weights = self.model.pwmlike_layer.weight
        stem_ks = self.tr_cfg.stem_ks
        with torch.no_grad():
            for pwm_idx, pwm_path in enumerate(pwm_paths):
                pwm_df = pd.read_csv(pwm_path, 
                                     sep=' ', 
                                     skiprows=[0], 
                                     names=['A', 'C', 'G', 'T'])
                pwm_len = len(pwm_df.index)
                
                if self.tr_cfg.pwm_loc == 'middle':
                    left = (stem_ks - pwm_len) // 2
                    right = left + pwm_len
                else:
                    left = 0
                    right = pwm_len
                
                pwmlike_weights[2 * pwm_idx, 0:4, :] = 0
                pwmlike_weights[2 * pwm_idx, 0:4, left:right] = \
                    get_weigths_from_pwm(pwm_df)
                pwmlike_weights[2 * pwm_idx + 1, 0:4, :] = 0
                pwmlike_weights[2 * pwm_idx + 1, 0:4, left:right] = \
                    get_weigths_from_pwm(pwm_df, rev=True, compl=True)
                    #stem_ks-right:stem_ks-left
        
        
        
    def training_step(self, batch, _):
        X, y = batch
        y_pred = self.model(X)

        loss = self.loss(y_pred, y)
        
        self.log("train_loss", 
                 loss, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.log('val_loss',
                 loss, 
                 on_step=False, 
                 on_epoch=True)
        y_prob = self.sigmoid(y_pred)
        self.metric(y_prob, y)
        self.log('val_' + self.metric_name, 
                 self.metric, 
                 on_epoch=True)
    
    def test_step(self, batch, _):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.log('test_loss', 
                 loss, 
                 prog_bar=True, 
                 on_step=False,
                 on_epoch=True)
        y_prob = self.sigmoid(y_pred)
        self.metric(y_prob, y)
        self.log('test_' + self.metric_name, 
                 self.metric, 
                 prog_bar=True, 
                 on_step=False,
                 on_epoch=True)
        
    def predict_step(self, batch, _):
        if isinstance(batch, (tuple, list)):
            x, _ = batch 
        else:
            x = batch
        y_pred = self.model(x)
        y_prob = self.sigmoid(y_pred)
        return y_prob
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.tr_cfg.max_lr  / 25,
                                      weight_decay=self.tr_cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, # type: ignore
                                                        max_lr=self.tr_cfg.max_lr ,
                                                        three_phase=False, 
                                                        total_steps=self.trainer.estimated_stepping_batches, # type: ignore
                                                        pct_start=0.3,
                                                        cycle_momentum =False)
        
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": "cycle_lr"
        }
        return [optimizer], [lr_scheduler_config]
