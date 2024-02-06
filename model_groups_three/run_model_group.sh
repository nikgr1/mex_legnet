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

log_dir=/home/nikgr/mex_legnet/model_groups_three_LOGS/${group_name%/}
group_cfgs_path=/home/nikgr/mex_legnet/model_groups_three/${group_name%/}

mkdir -p $log_dir
log_ver=$(ls -1 | wc -l)
let "log_ver++"

for cfg in $group_cfgs_path/*
do
    echo -e $cfg
    python /home/nikgr/mex_legnet/core.py --device $device @$cfg | tee -a $log_dir/${log_ver}.txt
done