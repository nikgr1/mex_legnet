while getopts g:d: flag
do
    case "${flag}" in
        d) device=${OPTARG};;
        g) group_name=${OPTARG};;
    esac
done
[ -z "$device" ] && exit
[ -z "$group_name" ] && exit

source /home/nikgr/miniconda3/etc/profile.d/conda.sh
conda activate legnet

group_cfgs_path=/home/nikgr/mex_legnet/model_groups/$group_name
for cfg in $group_cfgs_path/*
do
    echo -e $cfg
    python /home/nikgr/mex_legnet/core.py --device $device @$cfg
done