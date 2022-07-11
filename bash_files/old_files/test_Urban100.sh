python tta_main.py \
        --input_dir "/workspace/ssd1_2tb/nax_projects/super_resolution/dataset/Urban100/LR_bicubic/x2" \
        --gt_dir "/workspace/ssd1_2tb/nax_projects/super_resolution/dataset/Urban100/HR" \
        --output_dir "test_swinir_Urban100" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 1 \
        --lr_G_UP 0.00002\
        --lr_G_DN 0.001 \
        --test_only

