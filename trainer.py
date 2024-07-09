import lightning.pytorch as pl
import torch
import torch.nn as nn

from model import initialize_weights
from torchmetrics import AUROC, AveragePrecision

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
        # self.metrics = {
        #     'auprc': AveragePrecision(task="binary"),
        #     'auroc': AUROC(task="binary"),
        # }
        self.metric = AUROC(task="binary")
        self.metric_name = 'auroc'
        self.sigmoid = nn.Sigmoid()
    
    def calc_ic_vec(self, pwm_tensor):
        # Assume 'pwm_tensor' is a PyTorch tensor with shape (batch_size, 4, pwm_length)
        batch_size, num_nucleotides, pwm_length = pwm_tensor.shape
        
        shape = (batch_size, pwm_length)
        ic_vec = torch.zeros(shape, requires_grad = False)

        for i in range(batch_size):
            for j in range(pwm_length):
                nucleotide_types = pwm_tensor[i, 0:4, j].squeeze()  # (4,) tensor
                probs = torch.softmax(nucleotide_types, dim=0)  # probability distribution over the 4 nucleotide types
                ic = -torch.sum(probs * torch.log2(probs))  # calculate information content
                ic_vec[i, j] = 2 - ic.item()
        return ic_vec.to(self.tr_cfg.device)
    
    def calc_scale_vec(self, ic_vec, transform=None, shift=1):
        if transform is None:
            transform = nn.ReLU()
        tensor = transform(ic_vec - shift)
        tensor = torch.where(tensor > 0, tensor + shift, tensor)
        tensor.requires_grad = False
        return tensor.to(self.tr_cfg.device)
    
    def set_stem_requires_grad(self, requires_grad=None):
        if requires_grad is None:
            requires_grad = not self.tr_cfg.pwms_freeze
        print('<3> Unfreezing' if requires_grad else 'Freezing', 'stem conv layer', f'{requires_grad=}')
        for param in self.model.pwmlike_layer.parameters():
            param.requires_grad = requires_grad
    
    def initialize_weights(self):
        self.model.apply(initialize_weights)
        if self.tr_cfg.pwms_path is not None:
            self.initialize_stem_with_pwms()
        
    def initialize_stem_with_pwms(self):
        if self.tr_cfg.pwms_path is None:
            return
        pwms_path = Path(self.tr_cfg.pwms_path)
        pwm_paths = list(pwms_path.rglob('*/*.pwm'))
        pwm_count = len(pwm_paths)
        if pwm_count == 0:
            raise Exception('No PWMs were found')
        print(f'<1> Initializing stem conv layer with {pwm_count} PWMs')
        if self.tr_cfg.stem_ch//2 < pwm_count:
            print('<1> Amount of stem channels is less than number of PWMs')
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
            print('<2> Copy & freeze pwmlike_weights')
            self.goal_weights = pwmlike_weights.clone().detach().to(self.tr_cfg.device)
            self.goal_weights.requires_grad = False
            print('<3> Calculate scale tensor from information content')
            ic_tensor = self.calc_ic_vec(pwmlike_weights)
            self.scale_tensor = self.calc_scale_vec(ic_tensor, shift=1.5)
            # print(self.scale_tensor[0, :])
            print('<4> Done')
            # Testing
            # tmp = pwmlike_weights[:, 0, :]
            # pwmlike_weights[:, 0, :] = pwmlike_weights[:, 1, :]
            # pwmlike_weights[:, 1, :] = pwmlike_weights[:, 2, :]
            # pwmlike_weights[:, 2, :] = pwmlike_weights[:, 3, :]
            # pwmlike_weights[:, 3, :] = tmp
            # mse = self.calc_additional_loss()
            # print(mse)
    
    def calc_additional_loss(self):
        # MSE mean across nucleotides (batch_size, num_nucleotides, pwm_length) -> (batch_size, pwm_length)
        mse = ((self.model.pwmlike_layer.weight - self.goal_weights) ** 2).mean(dim=(1, ))
        # Scale the MSE by the scaling tensor
        scaled_mse = mse * self.scale_tensor
        return scaled_mse.mean()
        
        
        
    def training_step(self, batch, _):
        X, y = batch
        y_pred = self.model(X)

        loss = self.loss(y_pred, y)
        add_loss = self.calc_additional_loss() * 10_000
        sum_loss = loss + add_loss
        
        self.log("train_loss", 
                 loss, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True)
        self.log("train_add_loss", 
                 add_loss, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True)
        self.log("train_sum_loss", 
                 sum_loss, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True)
        return sum_loss
    
    def validation_step(self, batch, _):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.log('val_loss',
                 loss, 
                 on_step=False, 
                 on_epoch=True)
        y_prob = self.sigmoid(y_pred)
        # y_int = y.int()
        self.metric(y_prob, y)
        self.log('val_' + self.metric_name, 
                 self.metric, 
                 on_epoch=True)
        # for key, metric in self.metrics.items():
        #     value = metric(y_prob, y_int)
        #     self.log('val_' + key, 
        #              value, 
        #              on_epoch=True,
        #              on_step=False)
    
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
        y_int = y.int()
        # for key, metric in self.metrics.items():
        #     value = metric(y_prob, y_int)
        #     self.log('val_' + key, 
        #          metric, 
        #          on_epoch=True)
        self.metric(y_prob, y)
        self.log('val_' + self.metric_name, 
                 self.metric, 
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
