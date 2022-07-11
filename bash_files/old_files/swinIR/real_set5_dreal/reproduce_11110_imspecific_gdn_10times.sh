



strategy="11110"

CUDA_VISIBLE_DEVICES=1 python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/my_RealSR/Test/2_3_3/LR" \
        --gt_dir "../../dataset/my_RealSR/Test/2_3_3/HR" \
        --output_dir "reproduce_imspecific_gdn-lr_10times-"$strategy \
        --pretrained_gdn "random_init" \
        --training_strategy $strategy \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 10 \
        --lr_G_UP 0.00002\
        --lr_G_DN 0.001 \
        --lr_G_UP_step_size 1000\
        --g_input_shape 48 \
        --d_input_shape 24 \
        --finetune_gdn



