while getopts t:g: flag
do
    case "${flag}" in
        t) tfs=${OPTARG};;
        g) group_name=${OPTARG};;
    esac
done

source ../platform_imports.sh
mkdir -p $group_name

for tf in $tfs; do
    exp1=CHS
    exp2=GHTS
    pwms=$path_to_base/MEX/best_20_motif_CHS_GHTS
    for num in {1..2}; do
        args="--model_dir=$path_to_base/mex_models/$group_name/$tf-$exp1-$exp2
--train_path=$path_to_base/MEX/DATASETS/$exp1/Train/$tf
--valid_path=$path_to_base/MEX/DATASETS/$exp1/Test/$tf
--test_path=$path_to_base/MEX/DATASETS/$exp2/Test/$tf
--ref_genome_path=$path_to_base/hg38/hg38.fa
--stem_ks=40
--stem_ch=40
--epoch_num=20
--max_lr=0.0025
--reverse_augment
--use_shift
--max_shift 25 25
--num_workers 24
--negatives random
--pwms_path $pwms/$exp1/$tf
--pwms_freeze
--pwm_loc edge"
        echo -e "$args" > $group_name/$tf-$exp1-$exp2.cfg
        tmp=$exp1
        exp1=$exp2
        exp2=$tmp
    done
done

