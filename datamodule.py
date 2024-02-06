import lightning.pytorch as pl
import pandas as pd

from torch.utils.data import DataLoader
from dataset import TrainSeqDatasetProb, TestSeqDatasetProb

from training_config import TrainingConfig
from pathlib import Path
from Bio import SeqIO


splits = ('train', 'val', 'test')

def get_file_name(seq_type: str) -> str:    
    match seq_type:
        case 'positives':
            return 'positives.bed'
        case 'foreigns':
            return 'foreigns.bed'
        case 'random':
            return 'random_addshift.bed'
        case 'shades':
            return 'shades_addshift.bed'
        case 'shades_one':
            return 'shades_one.bed'
        case _:
            raise Exception('Wrong sequence type')

def get_seq_value(seq_type: str) -> int:
    match seq_type:
        case 'positives':
            return 1
        case _:
            return 0

class SeqDataModule(pl.LightningDataModule):
    def __init__(self,
                 cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg
        paths = (self.cfg.train_path, 
                 self.cfg.valid_path, 
                 self.cfg.test_path)
        self.paths = dict(zip(splits, paths))
        self.ds = {}
        
        
        dfs2concat = {split:list() for split in splits}
        columns = ['chr', 'start', 'end']
        for split, path in self.paths.items():
            if split == 'test':
                seq_types = ['positives'] + self.cfg.negatives_test
            else:
                seq_types = ['positives'] + self.cfg.negatives
            print(seq_types)
            for seq_type in seq_types:
                df = pd.read_csv(Path(path) / get_file_name(seq_type),
                                 usecols=range(3),
                                 sep='\t')
                df.columns = columns
                df['class_'] = get_seq_value(seq_type)
                dfs2concat[split].append(df)
            
            self.ds[split] = pd.concat(dfs2concat[split])
            
        self.ds_statistics()
        self.ref_genome = SeqIO.to_dict(SeqIO.parse(self.cfg.ref_genome_path, 'fasta'))
    
    def ds_statistics(self):
        print('Dataset statistics')
        for split, ds in self.ds.items():
            if split == 'test':
                seq_types = self.cfg.negatives
            else:
                seq_types = self.cfg.negatives_test
            count = len(ds['class_'])
            print('Split:', split, count, 'objects')
            print('Negatives:', ', '.join(seq_types))
            s = ds['class_'].value_counts()
            print('\t| '.join(f'{i}: {v} ({v*100/count:.2f} %)' for i, v in s.items()))
        
        
    def train_dataloader(self):
        
        train_ds =  TrainSeqDatasetProb(self.ds['train'],
                                   use_reverse=self.cfg.reverse_augment,
                                   use_reverse_channel=self.cfg.use_reverse_channel,
                                   use_shift=self.cfg.use_shift,
                                   max_shift=self.cfg.max_shift,
                                   ref_genome=self.ref_genome)
        
        return DataLoader(train_ds, 
                          batch_size=self.cfg.train_batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=True,
                          drop_last=True) 
    
    def val_dataloader(self):
        valid_ds = TestSeqDatasetProb(self.ds['val'], 
                                  use_reverse_channel=self.cfg.use_reverse_channel,
                                  shift=0,
                                  reverse=False,
                                  ref_genome=self.ref_genome)

        return DataLoader(valid_ds, 
                          batch_size=self.cfg.valid_batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=False,
                          drop_last=True)
        
    def dls_for_predictions(self):
        
        test_ds = TestSeqDatasetProb(self.ds['test'],
                                     use_reverse_channel=self.cfg.use_reverse_channel,
                                     shift=0,
                                     reverse=False,
                                     ref_genome=self.ref_genome)
        test_dl =  DataLoader(test_ds,
                              batch_size=self.cfg.valid_batch_size,
                              num_workers=self.cfg.num_workers,
                              shuffle=False,
                              drop_last=True)
        yield "forw_pred", test_dl
        if self.cfg.reverse_augment:
            rev_test_ds = TestSeqDatasetProb(self.ds['test'],
                                             use_reverse_channel=self.cfg.use_reverse_channel,
                                             shift=0,
                                             reverse=True,
                                             ref_genome=self.ref_genome)
            rev_test_dl =  DataLoader(rev_test_ds,
                                      batch_size=self.cfg.valid_batch_size,
                                      num_workers=self.cfg.num_workers,
                                      shuffle=False,
                                      drop_last=True)
            yield "rev_pred", rev_test_dl
