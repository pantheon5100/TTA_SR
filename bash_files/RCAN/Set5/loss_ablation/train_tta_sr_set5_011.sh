# python tta_main.py \
#         --input_dir "../../dataset/Set5/LR_bicubic/x2" \
#         --gt_dir "../../dataset/Set5/HR" \
#         --output_dir "TTA_set5-011" \
#         --training_strategy "011" \
#         --source_model "rcan" \
#         --num_iters 4000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 3000 \
#         --eval_iters 2 \
#         --lr_G_UP 0.000002\
#         --lr_G_DN 0.001

# GDN can not converge. try to decrease the learning rate.

# python tta_main.py \
#         --input_dir "../../dataset/Set5/LR_bicubic/x2" \
#         --gt_dir "../../dataset/Set5/HR" \
#         --output_dir "TTA_set5-011-lr_gdn_1e-4" \
#         --training_strategy "011" \
#         --source_model "rcan" \
#         --num_iters 4000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 3000 \
#         --eval_iters 2 \
#         --lr_G_UP 0.000002\
#         --lr_G_DN 0.0001

# and increase the learning rate
python tta_main.py \
        --input_dir "../../dataset/Set5/LR_bicubic/x2" \
        --gt_dir "../../dataset/Set5/HR" \
        --output_dir "TTA_set5-011-lr_gdn_1e-2" \
        --training_strategy "011" \
        --source_model "rcan" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.01
