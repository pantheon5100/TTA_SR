python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/Set14/LR_bicubic/x2" \
        --gt_dir "../../dataset/Set14/HR" \
        --output_dir "tta_main_train_unified_gdn_Set14-111" \
        --training_strategy "111" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001


# --pretrained_gdn "log/tta_main_train_unified_gdn_Set14-111/time_20220621051700lr_GUP_2e-06-lr_GDN_0.001input_size_48-scale_factor_2/ckpt/pretrained_GDN.ckpt" \
