while getopts c:s:g: flag
do
    case "${flag}" in
        c) tfs_chs=${OPTARG};;
        s) tfs_ghts=${OPTARG};;
        g) group_name=${OPTARG};;
    esac
done

source ../platform_imports.sh
mkdir -p $group_name
pwms=$path_to_base/MEX/best_20_motif_CHS_GHTS

for tf in $tfs_chs; do
    exp1=CHS
    exp2=GHTS
    first_arg="--model_dir=$path_to_base/mex_models_prod/$group_name/$tf-$exp1-$exp2"
    args1="--train_path=$path_to_base/MEX/DATASETS/$exp1/Train/$tf
--valid_path=$path_to_base/MEX/DATASETS/$exp1/Test/$tf
--test_path=$path_to_base/MEX/DATASETS/$exp2/Test/$tf
--ref_genome_path=$path_to_base/hg38/hg38.fa
--stem_ks=40
--stem_ch=40
--epoch_num=15
--max_lr=0.0025
--reverse_augment
--use_shift
--max_shift 25 25
--negatives random"
    echo -e "$first_arg-1
$args1" > $group_name/$tf-$exp1-$exp2-1.cfg

    args2="$args1
--pwms_path $pwms/$exp1/$tf
--pwms_freeze
--pwm_loc edge"
    echo -e "$first_arg-2
$args2" > $group_name/$tf-$exp1-$exp2-2.cfg

    args3="$args2
--model_type=PWMNet"
    echo -e "$first_arg-3
$args3" > $group_name/$tf-$exp1-$exp2-3.cfg

done

for tf in $tfs_ghts; do
    exp1=GHTS
    exp2=CHS
    first_arg="--model_dir=$path_to_base/mex_models_prod/$group_name/$tf-$exp1-$exp2"
    args1="--train_path=$path_to_base/MEX/DATASETS/$exp1/Train/$tf
--valid_path=$path_to_base/MEX/DATASETS/$exp1/Test/$tf
--test_path=$path_to_base/MEX/DATASETS/$exp2/Test/$tf
--ref_genome_path=$path_to_base/hg38/hg38.fa
--stem_ks=40
--stem_ch=40
--epoch_num=15
--max_lr=0.0025
--reverse_augment
--use_shift
--max_shift 25 25
--negatives random"
    echo -e "$first_arg-1
$args1" > $group_name/$tf-$exp1-$exp2-1.cfg

    args2="$args1
--pwms_path $pwms/$exp1/$tf
--pwms_freeze
--pwm_loc edge"
    echo -e "$first_arg-2
$args2" > $group_name/$tf-$exp1-$exp2-2.cfg

    args3="$args2
--model_type=PWMNet"
    echo -e "$first_arg-3
$args3" > $group_name/$tf-$exp1-$exp2-3.cfg
done