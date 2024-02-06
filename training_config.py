import json
import sys
import torch.nn as nn


from model import LegNet, PWMNet
from dataclasses import dataclass, asdict
from pathlib import Path
from argparse import ArgumentParser

@dataclass
class TrainingConfig: 
    stem_ch: int
    stem_ks: int
    ef_ks: int
    ef_block_sizes: list[int]
    resize_factor: int
    pool_sizes: list[int]
    reverse_augment: bool
    use_reverse_channel: bool
    use_shift: bool
    max_shift: tuple[int, int] | None
    max_lr: float
    weight_decay: float
    model_dir: str 
    train_path: str
    ref_genome_path: str
    valid_path: str
    test_path: str
    epoch_num: int 
    device: int  
    seed: int
    train_batch_size: int
    valid_batch_size: int
    num_workers: int
    training: bool
    negatives: list[str]
    negatives_test: list[str]
    pwms_path: str | None
    pwms_freeze: int
    pwm_loc: str
    model_type: str
    
    def __post_init__(self):
        self.check_params()
        model_dir = Path(self.model_dir)
        if self.training:
            model_dir.mkdir(exist_ok=True,
                            parents=True)
            self.dump()
        
            if self.pwms_path is not None:
                pwms_path = Path(self.pwms_path)
                pwm_paths = list(pwms_path.rglob('*/*.pwm'))
                pwm_count = len(pwm_paths)
                if pwm_count == 0:
                    raise Exception('No PWMs were found')
            
                self.stem_ch = pwm_count * 2
                print(f'Setting stem_ch to {self.stem_ch}, path: {self.pwms_path}')
        
    
    def check_params(self): 
        if Path(self.model_dir).exists():
            print(f"Warning: model dir already exists: {self.model_dir}", file=sys.stderr)
        if not self.reverse_augment:
            if self.use_reverse_channel:
                raise Exception("If model uses reverse channel"
                                "reverse augmentation must be performed")
        
           
    def dump(self, path: str | Path | None = None):
        if path is None:
            path = Path(self.model_dir) / "config.json"
        self.to_json(path)
        
    def to_dict(self) -> dict:
        dt = asdict(self)
        return dt
    
    def to_json(self, path: str | Path):
        dt = self.to_dict()
        with open(path, 'w') as out:
            json.dump(dt, out, indent=4)
  
    @classmethod
    def from_dict(cls, dt: dict, training: bool | None = None, exclude: set | list | tuple | None = None) -> 'TrainingConfig':
        config = dict(dt)
        print(f'{config["model_type"]} is selected')
        assert config['model_type'] == 'LegNet' or config['model_type'] == 'PWMNet'
        if training is not None:
            config['training'] = training
        if exclude is None:
            exclude = set()
        return cls(**{k: v for k, v in config.items() if k not in exclude})
    
          
    @classmethod
    def from_json(cls, path: Path | str, training: bool = False) -> 'TrainingConfig':
        with open(path, 'r') as inp:
            config = json.load(inp)
        assert config['model_type'] == 'LegNet' or config['model_type'] == 'PWMNet'
        config['training'] = training
        return cls.from_dict(config)
  
    @property
    def in_ch(self) -> int:
       return 4 + self.use_reverse_channel
  
    def get_model(self) -> nn.Module:
        if self.model_type == 'LegNet':
            return LegNet(in_ch=self.in_ch,
                stem_ch=self.stem_ch,
                stem_ks=self.stem_ks,
                ef_ks=self.ef_ks,
                ef_block_sizes=self.ef_block_sizes,  
                resize_factor=self.resize_factor,
                pool_sizes=self.pool_sizes)
        if self.model_type == 'PWMNet':
            return PWMNet(in_ch=self.in_ch,
                stem_ch=self.stem_ch,
                stem_ks=self.stem_ks)
    
    def switch_testing(self):
        self.training = not self.training
        
    def swap_val_test_paths(self) -> 'TrainingConfig':
        swapped = self.to_dict()
        swapped['valid_path'] = self.test_path
        swapped['test_path'] = self.valid_path
        return self.from_dict(swapped)
    
    def set_negatives_test(self, negatives) -> 'TrainingConfig':
        switched = self.to_dict()
        if isinstance(negatives, str):
            negatives = [negatives]
        switched['negatives_test'] = negatives
        return self.from_dict(switched)
    
    def print_info(self):
        print(f'{"TrainingConfig":-^64}')
        print(f'{"Train":16}', f'{self.train_path:48}', sep='')
        print(f'{"Valid":16}', f'{self.valid_path:48}', sep='')
        print(f'{"Test":16}', f'{self.test_path:48}', sep='')
        print(f'{"Negatives":16}', f'{"".join(self.negatives):48}', sep='')
        print(f'{"Negatives test":16}', f'{"".join(self.negatives_test):48}', sep='')
        print('-' * 64)