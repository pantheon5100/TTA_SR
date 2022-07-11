



python tta_main_train_unified_gdn.py \
        --input_dir "../../dataset/Set5/LR_bicubic/x2" \
        --gt_dir "../../dataset/Set5/HR" \
        --output_dir "test-classicalSR_s2-" \
        --pretrained_gdn "random_init" \
        --source_model "swinir" \
        --swinir_task "classicalSR_s2" \
        --test_only
