#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"


CUDA_LAUNCH_BLOCKING=1 python distillation.py \
    --default_root_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_12 \
    --cache_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/cache_dir \
    --output_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_12/output_dir \
    --data_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/xsum \
    --teacher google/pegasus-xsum \
    --student_decoder_layers 12 --student_encoder_layers 16 \
    --val_check_interval 0.1 --n_val 1000 --eval_beams 2 \
    --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
    --model_name_or_path IGNORED \
    --alpha_hid=3. \
    --train_batch_size=8 --eval_batch_size=4 --gradient_accumulation_steps=16 \
    --num_train_epochs=6 \
    --sortish_sampler \
    --weight_decay 0.01 \
    --auto_scale_batch_size power \
    --warmup_steps 500 \
    --learning_rate=1e-4 \
    --lr_scheduler cosine_w_restarts \
    --do_train \
    --do_predict \
    --weights_summary true \
    --max_source_length 512 \
    --gpus 1 \
    --freeze_embeds --freeze_encoder --label_smoothing 0.1 --adafactor --overwrite_output_dir\
    "$@"
    #--sortish_sampler \
    #--weight_decay 0.01 \
    #--auto_scale_batch_size power \


    CUDA_LAUNCH_BLOCKING=1 python distillation.py \
    --default_root_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_8 \
    --cache_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/cache_dir \
    --output_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_8/output_dir \
    --data_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/xsum \
    --teacher /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_12/output_dir/student \
    --student_decoder_layers 8 --student_encoder_layers 16 \
    --val_check_interval 0.1 --n_val 1000 --eval_beams 2 \
    --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
    --model_name_or_path IGNORED \
    --alpha_hid=3. \
    --train_batch_size=12 --eval_batch_size=4 --gradient_accumulation_steps=16 \
    --num_train_epochs=6 \
    --sortish_sampler \
    --weight_decay 0.01 \
    --auto_scale_batch_size power \
    --warmup_steps 500 \
    --learning_rate=1e-4 \
    --lr_scheduler cosine_w_restarts \
    --do_train \
    --do_predict \
    --weights_summary true \
    --max_source_length 512 \
    --gpus 1 \
    --freeze_embeds --freeze_encoder --label_smoothing 0.1 --adafactor --overwrite_output_dir\
    "$@"


    CUDA_LAUNCH_BLOCKING=1 python distillation.py \
    --default_root_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_4 \
    --cache_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/cache_dir \
    --output_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_4/output_dir \
    --data_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/xsum \
    --teacher /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_8/output_dir/student \
    --student_decoder_layers 4 --student_encoder_layers 16 \
    --val_check_interval 0.1 --n_val 1000 --eval_beams 2 \
    --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
    --model_name_or_path IGNORED \
    --alpha_hid=3. \
    --train_batch_size=16 --eval_batch_size=4 --gradient_accumulation_steps=16 \
    --num_train_epochs=6 \
    --sortish_sampler \
    --weight_decay 0.01 \
    --auto_scale_batch_size power \
    --warmup_steps 500 \
    --learning_rate=1e-4 \
    --lr_scheduler cosine_w_restarts \
    --do_train \
    --do_predict \
    --weights_summary true \
    --max_source_length 512 \
    --gpus 1 \
    --freeze_embeds --freeze_encoder --label_smoothing 0.1 --adafactor --overwrite_output_dir\
    "$@"