

python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/Set5/LR_bicubic/x2" \
        --gt_dir "../../dataset/Set5/HR" \
        --output_dir "tta_sr-patch_size_96-11111" \
        --pretrained_gdn "random_init" \
        --training_strategy 11111 \
        --num_iters 4000 \
        --input_crop_size 96 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --finetune_gdn \
        --pretrained_gdn_with_imgenet
