python tta_main.py \
        --input_dir "../../dataset/Set5/LR_bicubic/x4" \
        --gt_dir "../../dataset/Set5/HR" \
        --output_dir "TTA_set5-x4-011" \
        --training_strategy "011" \
        --num_iters 4000 \
        --input_crop_size 96 \
        --scale_factor 4 \
        --scale_factor_downsampler 0.25 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001

