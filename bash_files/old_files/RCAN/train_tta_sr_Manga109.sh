python tta_main.py \
        --input_dir "../../dataset/Manga109/LR_bicubic/x2" \
        --gt_dir "../../dataset/Manga109/HR" \
        --output_dir "RCAN-TTA_Manga109-gup_eval" \
        --source_model "rcan" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --test_only

