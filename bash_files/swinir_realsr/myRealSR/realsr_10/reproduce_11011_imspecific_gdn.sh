


strategy="11011"

CUDA_VISIBLE_DEVICES=4 python tta_main_train_unified_gdn.py \
        --input_dir "../dataset/my_RealSR/Test/2_10/LR" \
        --gt_dir "../dataset/my_RealSR/Test/2_10/HR" \
        --output_dir "reproduce_imspecific_gdn-"$strategy \
        --pretrained_gdn "random_init" \
        --training_strategy $strategy \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --finetune_gdn
