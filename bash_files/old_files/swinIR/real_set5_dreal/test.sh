


# strategy="11111"

# CUDA_VISIBLE_DEVICES=0 python tta_main_train_unified_gdn.py \
#         --input_dir "../../dataset/my_RealSR/Test/2_3_3/LR" \
#         --gt_dir "../../dataset/my_RealSR/Test/2_3_3/HR" \
#         --output_dir test \
#         --pretrained_gdn "random_init" \
#         --training_strategy $strategy \
#         --num_iters 4000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 3000 \
#         --eval_iters 2 \
#         --lr_G_UP 0.00002\
#         --lr_G_DN 0.001 \
#         --lr_G_UP_step_size 1000\
#         --g_input_shape 48 \
#         --d_input_shape 24 \
#         --finetune_gdn \
#         --test_only




strategy="11111"

CUDA_VISIBLE_DEVICES=0 python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/my_RealSR/Test/2_6/LR" \
        --gt_dir "../../dataset/my_RealSR/Test/2_6/HR" \
        --output_dir test \
        --pretrained_gdn "random_init" \
        --training_strategy $strategy \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.00002\
        --lr_G_DN 0.001 \
        --lr_G_UP_step_size 1000\
        --g_input_shape 48 \
        --d_input_shape 24 \
        --finetune_gdn \
        --test_only