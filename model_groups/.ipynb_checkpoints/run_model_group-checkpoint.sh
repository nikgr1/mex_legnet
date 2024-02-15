while getopts g:d: flag
do
    case "${flag}" in
        d) device=${OPTARG};;
        g) group_name=${OPTARG};;
    esac
done
[ -z "$device" ] && exit
[ -z "$group_name" ] && exit


source ../platform_imports.sh
conda init bash
conda activate legnet

echo 'TEST2>' $path_to_base '<'
pwd

group_cfgs_path=$path_to_base/mex_legnet/model_groups/${group_name%/}
for cfg in $group_cfgs_path/*
do
    echo -e $cfg
    python $path_to_base/mex_legnet/core.py --device $device @$cfg
done