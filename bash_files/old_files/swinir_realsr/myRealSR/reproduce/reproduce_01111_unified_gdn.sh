

python tta_main_train_unified_gdn.py \
        --input_dir "../../DualSR/test/my_RealSR/Test/2/LR" \
        --gt_dir "../../DualSR/test/my_RealSR/Test/2/HR" \
        --output_dir "reproduce_unified_gdn-01111-real_sr" \
        --pretrained_gdn "log/tta_main_train_unified_gdn-01001-swinir-Set5-01001/time_20220623103428lr_GUP_2e-06-lr_GDN_0.001input_size_48-scale_factor_2/ckpt/pretrained_GDN.ckpt" \
        --training_strategy 01111 \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 0.000002\
        --lr_G_DN 0.001 \
        --pretrained_gdn_with_imgenet

