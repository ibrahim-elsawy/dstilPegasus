#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

# From appendix C of paper https://arxiv.org/abs/1912.08777
# Set --gradient_accumulation_steps  so that effective batch size is 256 (2*128, 4*64, 8*32, 16*16)
python finetune.py \
    --default_root_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out \
    --cache_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/cache_dir \
    --output_dir /home/nlp_workstation/pegasus_ats/XSUM_PEGASUS/piped_out/output_dir \
    --data_dir xsum \
    --model_name_or_path google/pegasus-xsum \
    --learning_rate=1e-4 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.25 \
    --max_source_length 512 --max_target_length 56 \
    --gpus 1 \
    --freeze_embeds --label_smoothing 0.1 --adafactor --task summarization_xsum \
    "$@"
