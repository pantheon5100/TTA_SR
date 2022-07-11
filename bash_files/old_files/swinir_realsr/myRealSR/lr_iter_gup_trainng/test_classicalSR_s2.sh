



strategy="11011"

CUDA_VISIBLE_DEVICES=1 python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/my_RealSR/Test/2_10/LR" \
        --gt_dir "../../dataset/my_RealSR/Test/2_10/HR" \
        --output_dir "test-classicalSR_s2-" \
        --pretrained_gdn "random_init" \
        --source_model "swinir" \
        --swinir_task "classicalSR_s2" \
        --test_only
