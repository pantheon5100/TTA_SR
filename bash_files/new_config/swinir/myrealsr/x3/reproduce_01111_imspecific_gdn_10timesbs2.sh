



strategy="11001"

CUDA_VISIBLE_DEVICES=7 python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/my_RealSR/Test/3_3_3/LR" \
        --gt_dir "../../dataset/my_RealSR/Test/3_3_3/HR" \
        --output_dir "bs2-"$strategy \
        --pretrained_gdn "random_init" \
        --training_strategy $strategy \
        --gdn_iters 200 \
        --gup_iters 1000 \
        --batch_size 2 \
        --input_crop_size 24 \
        --scale_factor 3 \
        --switch_iters 3000 \
        --eval_iters 10 \
        --lr_G_UP 3.4e-6 \
        --lr_G_DN 1.1e-2 \
        --finetune_gdn



# # for debug
# strategy="11001"

# CUDA_VISIBLE_DEVICES=7 python tta_main_train_unified_gdn.py \
#         --input_dir "test_set/set5_x3/LR" \
#         --gt_dir "test_set/set5_x3/HR" \
#         --output_dir "bs2-"$strategy \
#         --pretrained_gdn "random_init" \
#         --training_strategy $strategy \
#         --gdn_iters 200 \
#         --gup_iters 1000 \
#         --batch_size 2 \
#         --input_crop_size 24 \
#         --scale_factor 3 \
#         --switch_iters 3000 \
#         --eval_iters 10 \
#         --lr_G_UP 3.4e-6 \
#         --lr_G_DN 1.1e-2 \
#         --finetune_gdn

