lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="resources/chatglm-6b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="datasets/pre_data"  # 填入数据集所在的文件夹路径
your_checkpopint_path="saved"  # 填入用来存储模型的路径
MAX_STEPS=8000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
date="0706"
MAX_SOURCE_LENGTH=1024

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

# Training Command
deepspeed --num_gpus=4 --master_port $MASTER_PORT run_lora.py \
    --deepspeed src/ft_chatglm_all/ds.config \
    --do_train \
    --train_file $your_data_path/train_aug_2.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/chatglm-lora-$date \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps ${MAX_STEPS} \
    --logging_steps 100 \
    --save_steps 2000 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16

# Testing Command
deepspeed --num_gpus=2 --master_port $MASTER_PORT run_lora.py \
    --do_predict \
    --test_file $your_data_path/testB.json \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path $your_checkpopint_path/chatglm-lora-$date/checkpoint-$MAX_STEPS \
    --output_dir submission/lora \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate