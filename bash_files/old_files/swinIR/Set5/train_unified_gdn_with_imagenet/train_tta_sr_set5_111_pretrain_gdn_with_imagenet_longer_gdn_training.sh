python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/Set5/LR_bicubic/x2" \
        --gt_dir "../../dataset/Set5/HR" \
        --output_dir "tta_main_train_unified_gdn-111-longer_gdn_training6000-bs16" \
        --training_strategy 111 \
        --num_iters 4000 \
        --pretrained_gdn_num_iters 6000 \
        --input_crop_size 48 \
        --each_batch_img_size 16 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --pretrained_gdn_with_imgenet


# python tta_main_train_unified_gdn.py \
#         --input_dir "../../dataset/Set5/LR_bicubic/x2" \
#         --gt_dir "../../dataset/Set5/HR" \
#         --output_dir "tta_main_train_unified_gdn-111-longer_gdn_training6000" \
#         --training_strategy 111 \
#         --num_iters 4000 \
#         --pretrained_gdn_num_iters 6000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 3000 \
#         --eval_iters 2 \
#         --lr_G_UP 0.000002\
#         --lr_G_DN 0.001 \
#         --pretrained_gdn_with_imgenet
