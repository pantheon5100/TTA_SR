

TRAINING_STRATEGY="000 001 010 011 100 101 110 111"
for training_strategy in $TRAINING_STRATEGY
do
python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/Set5/LR_bicubic/x2" \
        --gt_dir "../../dataset/Set5/HR" \
        --output_dir "pretrained_gdn_with_imgenet-tta_main_train_unified_gdn-"$training_strategy \
        --source_model "rcan" \
        --training_strategy $training_strategy \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --pretrained_gdn_with_imgenet \
        --pretrained_gdn_only

done
