

TRAINING_STRATEGY="000 001 010 011 100 101 110 111"
for training_strategy in $TRAINING_STRATEGY
do
python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/Urban100/LR_bicubic/x2" \
        --gt_dir "../../dataset/Urban100/HR" \
        --output_dir "tta_main_train_unified_gdn-"$training_strategy \
        --pretrained_gdn "log/swinir-tta_main_train_unified_gdn-011/time_20220622075416lr_GUP_2e-06-lr_GDN_0.001input_size_48-scale_factor_2/ckpt/pretrained_GDN.ckpt" \
        --training_strategy $training_strategy \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --pretrained_gdn_with_imgenet

done
