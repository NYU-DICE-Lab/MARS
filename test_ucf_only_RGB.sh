model_path="/home/mp5847/src/MARS/results/UCF-101_only_RGB_sd=1.0-weight_decay=0/UCF101/PreKin_UCF101_1_RGB_train_batch64_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR290_sigma=1.0.pth"
frame_dir="/data/mp5847_dataset/UCF-101_extracted_frames"
annotation_path="./annotation/ucfTrainTestlist"
result_path="./results/UCF-101_only_RGB_sd=0.5"
python test_single_stream.py --batch_size 1 --n_classes 101 --model resnext --model_depth 101 --log 0 --dataset UCF101 --modality RGB --sample_duration 64 --split 1 --only_RGB --resume_path1 "${model_path}" --frame_dir "${frame_dir}" --annotation_path "${annotation_path}" --result_path "${result_path}" --noise_sd 1.0 --normalize_layer 1 --noise_augment 1 --n_workers 10 --test_mode 1 --overlapping 1