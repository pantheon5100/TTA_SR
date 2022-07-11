


# strategy="11011"

# CUDA_VISIBLE_DEVICES=6 python tta_main_train_unified_gdn.py \
#         --input_dir "../dataset/my_RealSR/Test/2_10/LR" \
#         --gt_dir "../dataset/my_RealSR/Test/2_10/HR" \
#         --output_dir "reproduce_imspecific_gdn-lr_4times-"$strategy \
#         --pretrained_gdn "random_init" \
#         --training_strategy $strategy \
#         --num_iters 4000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 3000 \
#         --eval_iters 2 \
#         --lr_G_UP 0.000008\
#         --lr_G_DN 0.001 \
#         --lr_G_UP_step_size 1000\
#         --finetune_gdn


strategy="01001"

CUDA_VISIBLE_DEVICES=1 python tta_main_train_unified_gdn.py \
        --input_dir "../dataset/my_RealSR/Test/2_10/LR" \
        --gt_dir "../dataset/my_RealSR/Test/2_10/HR" \
        --output_dir "reproduce_imspecific_gdn-lr_10times-"$strategy \
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
        --finetune_gdn
