import torch 

import lightning.pytorch as pl



from datamodule import SeqDataModule
from test_predict import save_predict
from trainer import LitModel, TrainingConfig
from utils import set_global_seed, parameter_count, ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path 
from Bio import SeqIO



# import argparse 
parser = ArgumentParser(fromfile_prefix_chars='@')
# parser.convert_arg_line_to_args = convert_arg_line_to_args

general = parser.add_argument_group('general args', 
                                    'general_argumens')
general.add_argument("--model_dir",
                     type=str,
                     required=True)
general.add_argument("--device", 
                     type=int,
                     default=0)
general.add_argument("--num_workers",
                     type=int, 
                     default=8)
general.add_argument("--seed",
                     type=int,
                     default=777)

aug = parser.add_argument_group('aug args', 
                                'augmentation arguments')
aug.add_argument("--reverse_augment", 
                 action="store_true")
aug.add_argument("--use_reverse_channel",
                 action="store_true")
aug.add_argument("--use_shift", 
                 action="store_true")
aug.add_argument("--max_shift",
                 default=None, 
                 nargs=2,
                 type=int)

model_args =  parser.add_argument_group('model arguments', 
                                        'model architecture arguments')
model_args.add_argument("--stem_ch", 
                        type=int,
                        default=64)
model_args.add_argument("--stem_ks",
                        type=int,
                        default=11)
model_args.add_argument("--ef_ks",
                        type=int,
                        default=9)
model_args.add_argument("--ef_block_sizes", 
                        type=int,
                        nargs="+",
                        default=[80, 96, 112, 128])
model_args.add_argument("--resize_factor",
                        type=int,
                        default=4)
model_args.add_argument("--pool_sizes", 
                        type=int,
                        nargs="+",
                        default=[2, 2, 2, 2])

scheduler_args =  parser.add_argument_group('scheduler arguments', 
                                            'One cycle scheduler arguments')
scheduler_args.add_argument("--max_lr", 
                            type=float,
                            default=0.01)
scheduler_args.add_argument("--weight_decay",
                            type=float,
                            default=0.1)
scheduler_args.add_argument("--epoch_num",
                            type=int,
                            default=20)

data_args =  parser.add_argument_group('data arguments',
                                        'Data arguments')
data_args.add_argument("--valid_batch_size",
                       type=int,
                       default=1024)
data_args.add_argument("--valid_path", 
                       type=str, 
                       required=True)
data_args.add_argument("--train_path", 
                       type=str, 
                       required=True)
data_args.add_argument("--train_batch_size",
                       type=int, 
                       default=1024)
data_args.add_argument("--test_path", 
                       type=str, 
                       required=True)
data_args.add_argument("--ref_genome_path", 
                       type=str, 
                       required=True)
data_args.add_argument("--lr_plot_path",
                       type=str,
                       required=True)


args = parser.parse_args()
vars_args = vars(args)
lr_plot_path = vars_args.pop('lr_plot_path', None)

train_cfg = TrainingConfig.from_dict(vars_args, training=True)
print(train_cfg)

model_dir = Path(train_cfg.model_dir)
model_dir.mkdir(exist_ok=True,
                parents=True)

train_cfg.dump()

torch.set_float32_matmul_precision('medium') # type: ignore 


model = LitModel(tr_cfg=train_cfg)
print(parameter_count(model))

data = SeqDataModule(cfg=train_cfg)

train_dl = data.train_dataloader()
valid_dl = data.val_dataloader()

dump_dir = model_dir / "model"
trainer = pl.Trainer(accelerator='gpu',
                     devices=[train_cfg.device], 
                     precision='16-mixed',
                     gradient_clip_val=1,
                     default_root_dir=dump_dir)

tuner = pl.tuner.Tuner(trainer)

# Run learning rate finder
lr_finder = tuner.lr_find(model,
                          train_dataloaders=train_dl,
                          attr_name='max_lr')

# Plot with
fig = lr_finder.plot(suggest=True)
fig.savefig(lr_plot_path, dpi=300)

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
print('Suggested lr:', new_lr)