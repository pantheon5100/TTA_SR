python tta_main.py \
        --input_dir "../../DualSR/test/my_RealSR/Test/2/LR" \
        --gt_dir "../../DualSR/test/my_RealSR/Test/2/HR" \
        --output_dir "EDSR-TTA_Manga109-gup_eval" \
        --source_model "edsr" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --test_only

