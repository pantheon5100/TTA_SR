python tta_main.py \
        --input_dir "../../dataset/Set5/LR_bicubic/x2" \
        --gt_dir "../../dataset/Set5/HR" \
        --output_dir "TTA_set5_bicubic" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001

