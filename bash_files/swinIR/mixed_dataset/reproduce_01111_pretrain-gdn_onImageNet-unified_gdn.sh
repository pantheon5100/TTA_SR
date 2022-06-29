

python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/mixed_dataset_3_3/LR_bicubic/x2" \
        --gt_dir "../../dataset/mixed_dataset_3_3/HR" \
        --output_dir "unified_gdn_onImageNet-01111" \
        --training_strategy 01111 \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --pretrained_gdn_with_imgenet

 