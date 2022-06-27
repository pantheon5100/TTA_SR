
python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/Set5/LR_bicubic/x2" \
        --gt_dir "../../dataset/Set5/HR" \
        --output_dir "tta_main_train_unified_gdn-101" \
        --pretrained_gdn "log/swinir-Set5-pretrained_gdn_with_imgenet-tta_main_train_unified_gdn-100-100/time_20220622161347lr_GUP_2e-06-lr_GDN_0.001input_size_48-scale_factor_2/ckpt/pretrained_GDN.ckpt" \
        --training_strategy 101 \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --pretrained_gdn_with_imgenet