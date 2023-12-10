while getopts t:g: flag
do
    case "${flag}" in
        t) tfs=${OPTARG};;
        g) group_name=${OPTARG};;
    esac
done

mkdir -p $group_name

for tf in $tfs; do
    exp1=CHS
    exp2=GHTS
    pwms1=/home/nikgr/MEX/best_20_motif_CHS_GHTS/CHS/LEUTX/THC_0312.Rep-DIANA_0293
    pwms2=/home/nikgr/MEX/best_20_motif_CHS_GHTS/GHTS/LEUTX/YWK_B_AffSeq_B7_LEUTX
    for num in {1..2}; do
        args="--model_dir=/home/nikgr/mex_models/$group_name/$tf-$exp1-$exp2
--train_path=/home/nikgr/MEX/DATASETS/$exp1/Train/$tf
--valid_path=/home/nikgr/MEX/DATASETS/$exp1/Test/$tf
--test_path=/home/nikgr/MEX/DATASETS/$exp2/Test/$tf
--ref_genome_path=/home/nikgr/hg38/hg38.fa
--stem_ks=40
--stem_ch=40
--epoch_num=20
--max_lr=0.0025
--reverse_augment
--use_reverse_channel
--use_shift
--max_shift 25 25
--num_workers 24
--negatives foreigns
--pwms_path $pwms1
--pwms_freeze
--pwm_loc edge"
        echo -e "$args" > $group_name/$tf-$exp1-$exp2.cfg
        tmp=$exp1
        exp1=$exp2
        exp2=$tmp
        tmp=$pwms1
        pwms1=$pwms2
        pwms2=$tmp
    done
done