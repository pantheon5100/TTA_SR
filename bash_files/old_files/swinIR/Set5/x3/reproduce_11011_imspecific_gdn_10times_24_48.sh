



strategy="11011"
g_input_shape=96
d_input_shape=32

CUDA_VISIBLE_DEVICES=1 python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/Set5/LR_bicubic/x3" \
        --gt_dir "../../dataset/Set5/HR" \
        --output_dir "reproduce_imspecific_gdn-2e_5-"$d_input_shape"_"$g_input_shape-$strategy \
        --pretrained_gdn "random_init" \
        --source_model "swinir" \
        --swinir_task "classicalSR_s1" \
        --training_strategy $strategy \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 3 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.00002\
        --lr_G_DN 0.001 \
        --lr_G_UP_step_size 1000\
        --g_input_shape $g_input_shape \
        --d_input_shape $d_input_shape \
        --finetune_gdn

