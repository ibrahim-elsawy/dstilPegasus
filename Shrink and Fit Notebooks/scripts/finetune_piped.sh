#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

#cd /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out
#wget https://cdn-datasets.huggingface.co/summarization/xsum.tar.gz
#tar -xzvf xsum.tar.gz
#cd /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/transformers/examples/research_projects/seq2seq-distillation


# From appendix C of paper https://arxiv.org/abs/1912.08777
# Set --gradient_accumulation_steps  so that effective batch size is 256 (2*128, 4*64, 8*32, 16*16)
# python distillation.py \
    # --default_root_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_12 \
    # --cache_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_12/cache_dir \
    # --output_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_12/output_dir \
    # --data_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/xsum \
    # --teacher google/pegasus-xsum \
    # --student_decoder_layers 12 --student_encoder_layers 16 \
    # --val_check_interval 0.1 --n_val 1000 --eval_beams 2 --length_penalty=0.5 \
    # --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
    # --model_name_or_path IGNORED \
    # --alpha_hid=0.2 \
    # --alpha_mlm=0.8 \
    # --alpha_ce=0.8 \
    # --train_batch_size=8 --eval_batch_size=4 --gradient_accumulation_steps=8 \
    # --sortish_sampler \
    # --num_train_epochs=6 \
    # --warmup_steps 500 \
    # --lr_scheduler cosine_w_restarts \
    # --weight_decay 0.01 \
    # --do_train \
    # --do_predict \
    # --val_metric loss\
    # --logger_name wandb \
    # --weights_summary true \
    # --auto_scale_batch_size true \
    # --max_source_length 512\
    # --gpus 1 \
    # --overwrite_output_dir \
    # --freeze_embeds --freeze_encoder --label_smoothing 0.1 --adafactor --task summarization_xsum_16_12 \
    # "$@"

CUDA_LAUNCH_BLOCKING=1 python distillation.py \
    --default_root_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_8 \
    --cache_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_8/cache_dir \
    --output_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_8/output_dir \
    --data_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/xsum \
    --teacher /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_12/output_dir/best_tfmr \
    --student_decoder_layers 8 --student_encoder_layers 16 \
    --val_check_interval 0.25 --n_val 1000 --eval_beams 2 --length_penalty=0.5 \
    --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
    --model_name_or_path IGNORED \
    --alpha_hid=0.2 \
    --alpha_mlm=0.8 \
    --alpha_ce=0.8 \
    --train_batch_size=8 --eval_batch_size=4 --gradient_accumulation_steps=32 \
    --sortish_sampler \
    --num_train_epochs=3 \
    --warmup_steps 500 \
    --lr_scheduler cosine_w_restarts \
    --weight_decay 0.01 \
    --do_train \
    --do_predict \
    --val_metric loss\
    --weights_summary true \
    --auto_scale_batch_size true \
    --max_source_length 512\
    --gpus 1 \
    --logger_name wandb \
    --overwrite_output_dir \
    --freeze_embeds --freeze_encoder --label_smoothing 0.1 --adafactor --task summarization_xsum_16_8 \
    "$@"

python distillation.py \
    --default_root_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_4 \
    --cache_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_4/cache_dir \
    --output_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_4/output_dir \
    --data_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/xsum \
    --teacher /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_students/xsum_16_8/output_dir/best_tfmr \
    --student_decoder_layers 4 --student_encoder_layers 16 \
    --val_check_interval 0.25 --n_val 1000 --eval_beams 2 --length_penalty=0.5 \
    --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
    --model_name_or_path IGNORED \
    --alpha_hid=0.2 \
    --alpha_mlm=0.8 \
    --alpha_ce=0.8 \
    --train_batch_size=8 --eval_batch_size=4 --gradient_accumulation_steps=32 \
    --sortish_sampler \
    --num_train_epochs=3 \
    --warmup_steps 500 \
    --lr_scheduler cosine_w_restarts \
    --weight_decay 0.01 \
    --do_train \
    --do_predict \
    --val_metric loss\
    --weights_summary true \
    --auto_scale_batch_size true \
    --max_source_length 512\
    --gpus 1 \
    --logger_name wandb \
    --overwrite_output_dir \
    --freeze_embeds --freeze_encoder --label_smoothing 0.1 --adafactor --task summarization_xsum_16_4 \
    "$@"

