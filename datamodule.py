import lightning.pytorch as pl
import pandas as pd

from torch.utils.data import DataLoader
from dataset import TrainSeqDatasetProb, TestSeqDatasetProb

from training_config import TrainingConfig
from pathlib import Path
from Bio import SeqIO

class SeqDataModule(pl.LightningDataModule):
    def __init__(self,
                 cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg
        splits = ('train', 'val', 'test')
        paths = (self.cfg.train_path, 
                 self.cfg.valid_path, 
                 self.cfg.test_path)
        self.paths = dict(zip(splits, paths))
        self.ds = {}
        
        vals_by_seq_types = {'foreigns': 0, 'positives': 1}
        dfs2concat = {split:list() for split in self.splits}
        columns = ['chr', 'start', 'end']
        for split, path in self.paths.items():
            for seq_type, value in vals_by_seq_types.items():
                df = pd.read_csv(Path(path) / (seq_type + '.bed'),
                                 usecols=range(3),
                                 sep='\t')
                df.columns = columns
                df['class_'] = value
                dfs2concat[split].append(df)
            
            self.ds[split] = pd.concat(dfs2concat[split])
            
        self.ds_statistics()
        self.ref_genome = SeqIO.to_dict(SeqIO.parse(self.cfg.ref_genome_path, 'fasta'))
    
    def ds_statistics(self):
        print('Dataset statistics')
        for split, ds in self.ds.items():
            print('Split:', split)
            s = ds['class_'].value_counts() / len(ds['class_'])
            print('\t'.join(f'{i}: {v:.2f}' for i, v in s.items()))
        
        
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
