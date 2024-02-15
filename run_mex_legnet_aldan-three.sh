#!/bin/bash -l

#SBATCH --job-name=MEX_LegNet               # Job name
#SBATCH --cpus-per-task=32                  # Run on a single CPU
#SBATCH --mem=32gb                          # Job memory request
#SBATCH --time=30-00:00:00                     # Time limit hrs:min:sec
#SBATCH --output=output/MEX_LegNet.%j.log   # Standard output and error log
#SBATCH --partition=infinite
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

group_name=finalize_run
device=0




source platform_imports.sh
cd model_groups
# bash run_model_group.sh -d 0 -g aldan_testing

conda activate legnet

group_cfgs_path=$path_to_base/mex_legnet/model_groups_three/${group_name%/}

for cfg in $group_cfgs_path/*
do
    echo -e $cfg
    python $path_to_base/mex_legnet/core.py --num_workers 32 --device $device @$cfg
done