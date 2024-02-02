import pandas as pd
import numpy as np
import pandas as pd
from Bio import SeqRecord

import torch

from torch.utils.data import  Dataset
from utils import Seq2Tensor, reverse_complement

class TrainSeqDatasetProb(Dataset):
    
    """ Sequence dataset. """
    
    def __init__(self, 
                 ds: pd.DataFrame,
                 use_reverse: bool,
                 use_shift: bool,
                 use_reverse_channel: bool,
                 ref_genome: dict[str, SeqRecord],
                 seqsize: int | None = None,
                 max_shift: tuple[int, int] | None = None, 
                 training=True):
        """
        Parameters
        ----------
        ds : pd.DataFrame
            Training dataset.
        use_reverse_channel : bool
            If True, additional reverse augmentation is used.
        seqsize : int
            Constant sequence length.
        """
        if seqsize is None:
            seqsize = abs(ds.iloc[0]['end'] - ds.iloc[0]['start'])
        if max_shift is None:
            max_shift = (0, 0)
        
        self.training = training

        self.ds = ds
        self.totensor = Seq2Tensor() 
        self.use_reverse = use_reverse
        self.use_shift = use_shift
        self.use_reverse_channel = use_reverse_channel
        self.ref_genome = ref_genome
        self.seqsize = seqsize
        self.max_shift = max_shift
            
    def transform(self, x):
        assert isinstance(x, str)
        return self.totensor(x)
    
    def __getitem__(self, i):
        entry = self.ds.iloc[i]
        chrom = entry.chr
        start = entry.start
        end = entry.end
        
        if self.use_shift:
            # we need to determine such max&min values for shift that it won't ruin the slicing step
            lowest = max(0, start-self.max_shift[0]) - start
            highest = min(len(self.ref_genome[chrom]), end+self.max_shift[1]) - end
            # calc shift
            shift = torch.randint(size=(1,), low=lowest, high=highest + 1).item()
            # do shift
            start = start + shift
            end = end + shift
        # slice
        seq = str(self.ref_genome[chrom][start:end].seq).upper()

        if self.use_reverse:
            r = torch.rand((1,)).item()
            if  r > 0.5:
                seq = reverse_complement(seq)
                rev = 1.0
            else:
                rev = 0.0
        else:
            rev = 0.0
            
        seq = self.transform(seq)
        to_concat = [seq]
        
        # add reverse augmentation channel
        if self.use_reverse_channel:
            rev = torch.full( (1, self.seqsize), rev, dtype=torch.float32)
            to_concat.append(rev)
            
        # create final tensor
        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq
            
        class_ = self.ds.class_.values[i]
        
        return X, class_.astype(np.float32)
    
    def __len__(self):
        return len(self.ds.start)
    
    
class TestSeqDatasetProb(Dataset):
    
    """ Sequence dataset. """
    
    def __init__(self, 
                 ds: pd.DataFrame,
                 reverse: bool,
                 shift: int,  
                 ref_genome: dict[str, SeqRecord],
                 seqsize: int | None = None,
                 use_reverse_channel: bool = True):
        """
        Parameters
        ----------
        ds : pd.DataFrame
            Training dataset.
        use_reverse_channel : bool
            If True, additional reverse augmentation is used.
        seqsize : int
            Constant sequence length.
        """
        if seqsize is None:
            seqsize = abs(ds.iloc[0]['end'] - ds.iloc[0]['start'])
       
        self.ds = ds
        self.totensor = Seq2Tensor()
        self.use_reverse_channel = use_reverse_channel 
        self.reverse = reverse
        self.shift = shift
        self.ref_genome = ref_genome
        self.seqsize = seqsize

        
    def transform(self, x):
        assert isinstance(x, str)
        return self.totensor(x)
    
    def __getitem__(self, i):
        """
        Output
        ----------
        X: torch.Tensor    
            Create one-hot encoding tensor with reverse and singleton channels if required.
        probs: np.ndarray
            Given a measured expression, we assume that the real expression is normally distributed
            with mean=`bin` and sd=`shift`. 
            Resulting `probs` vector contains probabilities that correspond to each class (bin).     
        bin: float 
            Training expression value
        """
        entry = self.ds.iloc[i]
        chrom = entry.chr
        start = entry.start
        end = entry.end
        # we need to use such shift that it won't ruin the slicing step
        shift = max(0, start+self.shift) - start
        shift = min(len(self.ref_genome[chrom]), end+shift) - end
        # do shift
        start = start + shift
        end = end + shift
        # slice
        seq = str(self.ref_genome[chrom][start:end].seq).upper()

        
        if self.reverse:
            seq = reverse_complement(seq)
            rev = 1.0
        else:
            rev = 0.0

        seq = self.transform(seq)
        to_concat = [seq]
        
        # add reverse augmentation channel
        if self.use_reverse_channel:
            rev = torch.full( (1, self.seqsize), rev, dtype=torch.float32)
            to_concat.append(rev)
            
        # create final tensor
        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq
            
        class_ = self.ds.class_.values[i]
        
        return X, class_.astype(np.float32)
    
    def __len__(self):
        return len(self.ds.start)


