import lightning.pytorch as pl
import pandas as pd

from torch.utils.data import DataLoader
from dataset import TrainSeqDatasetProb, TestSeqDatasetProb

from training_config import TrainingConfig
from pathlib import Path

class SeqDataModule(pl.LightningDataModule):
    def __init__(self,
                 cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg
        
        vals_by_seq_types = {'foreigns': 0, 'positives': 1}
        paths_by_splits = {'train': self.cfg.train_path, 
                           'val': self.cfg.valid_path,
                           'test': self.cfg.test_path}
        dfs = {k:list() for k in paths_by_splits.keys()}
        
        for split in paths_by_splits.keys():
            for seq_type in vals_by_seq_types.keys():
                df = pd.read_csv(Path(paths_by_splits[split]) / (seq_type + '.bed'),
                                 sep='\t')
                df.columns = ['chr', 'start', 'end']
                df['class'] = vals_by_seq_types[seq_type]
                dfs[split].append(df)
        
        self.train = pd.concat(dfs['train'])
        self.valid = pd.concat(dfs['val'])
        self.test = pd.concat(dfs['test'])
        self.ref_genome = SeqIO.to_dict(train_cfg.ref_genome_path, 'fasta')
        
        
    def train_dataloader(self):
        
        train_ds =  TrainSeqDatasetProb(self.train,
                                   use_reverse=self.cfg.reverse_augment,
                                   use_reverse_channel=self.cfg.use_reverse_channel,
                                   use_shift=self.cfg.use_shift,
                                   max_shift=self.cfg.max_shift,
                                   ref_genome=self.ref_genome)
        
        return DataLoader(train_ds, 
                          batch_size=self.cfg.train_batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=True) 
    
    def val_dataloader(self):
        valid_ds = TestSeqDatasetProb(self.valid, 
                                  use_reverse_channel=self.cfg.use_reverse_channel,
                                  shift=0,
                                  reverse=False,
                                  ref_genome=self.ref_genome)

        return DataLoader(valid_ds, 
                          batch_size=self.cfg.valid_batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=False)
        
    def dls_for_predictions(self):
        
        test_ds = TestSeqDatasetProb(self.test,
                                  use_reverse_channel=self.cfg.use_reverse_channel,
                                  shift=0,
                                  reverse=False,
                                  ref_genome=self.ref_genome)
        test_dl =  DataLoader(test_ds,
                              batch_size=self.cfg.valid_batch_size,
                              num_workers=self.cfg.num_workers,
                              shuffle=False)
        yield "forw_pred", test_dl
        if self.cfg.reverse_augment:
            rev_test_ds = TestSeqDatasetProb(self.test,
                                  use_reverse_channel=self.cfg.use_reverse_channel,
                                  shift=0,
                                  reverse=True,
                                  ref_genome=self.ref_genome)
            rev_test_dl =  DataLoader(rev_test_ds,
                              batch_size=self.cfg.valid_batch_size,
                              num_workers=self.cfg.num_workers,
                              shuffle=False)
            yield "rev_pred", rev_test_dl
