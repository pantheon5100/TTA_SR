
strategy="11111"

python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/my_RealSR/Test/2_10/LR" \
        --gt_dir "../../dataset/my_RealSR/Test/2_10/HR" \
        --output_dir "unified_gdn_onImageNet-"$strategy \
        --source_model "cdc" \
        --training_strategy $strategy \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --lr_G_UP_step_size 1000\
        --g_input_shape 108 \
        --d_input_shape 48 \
        --pretrained_gdn_with_imgenet

