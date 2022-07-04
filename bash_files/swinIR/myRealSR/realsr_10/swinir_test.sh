




# python tta_main_train_unified_gdn.py \
#         --input_dir "../dataset/my_RealSR/Test/2_10/LR" \
#         --gt_dir "../dataset/my_RealSR/Test/2_10/HR" \
#         --output_dir "test_swinir" \
#         --pretrained_gdn "random_init" \
#         --num_iters 4000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 3000 \
#         --eval_iters 2 \
#         --lr_G_UP 0.000002\
#         --lr_G_DN 0.001 \
#         --test_only

python tta_main_train_unified_gdn.py \
        --input_dir "../dataset/my_RealSR/Test/2/LR" \
        --gt_dir "../dataset/my_RealSR/Test/2/HR" \
        --output_dir "test_swinir" \
        --pretrained_gdn "random_init" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --test_only
