#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

CUDA_LAUNCH_BLOCKING=1 
python distillation.py \
    --default_root_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_12_4 \
    --cache_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_12_4/cache_dir \
    --output_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_12_4/output_dir \
    --data_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/xsum \
    --teacher google/pegasus-xsum \
    --student_decoder_layers 4 --student_encoder_layers 12 \
    --val_check_interval 0.25 --n_val 1000 --eval_beams 2 --length_penalty=0.5 \
    --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
    --model_name_or_path IGNORED \
    --alpha_hid=0\
    --alpha_mlm=1 \
    --alpha_ce=1 \
    --train_batch_size=8 --eval_batch_size=4 --gradient_accumulation_steps=32 \
    --sortish_sampler \
    --num_train_epochs=3 \
    --warmup_steps 500 \
    --lr_scheduler cosine_w_restarts \
    --weight_decay 0.01 \
    --do_train \
    --do_predict \
    --val_metric loss\
    --logger_name wandb \
    --weights_summary true \
    --auto_scale_batch_size true \
    --max_source_length 512\
    --gpus 1 \
    --overwrite_output_dir \
    --freeze_embeds --freeze_encoder --label_smoothing 0.1 --adafactor --task summarization_xsum_12_4 \
    "$@"