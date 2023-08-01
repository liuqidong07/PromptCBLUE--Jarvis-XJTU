lora_rank=2
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
data_path="datasets/pre_data/partition"  # 填入数据集所在的文件夹路径
checkpoint_path="saved"  # 填入用来存储模型的路径
glm_model_path="resources/chatglm-6b"
MAX_STEPS=500   # 设置最大的epoch数量
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
date="defualt"

task_list=('CMeEE-V2' 'IMCS-V2-DAC' 'CHIP-STS' 'MedDG' 'IMCS-V2-NER' 'CHIP-MDCFNPC' 'KUAKE-QTR' 'KUAKE-QIC' 'IMCS-V2-SR' 'KUAKE-QQR')

for task in ${task_list[@]}
do
    deepspeed --num_gpus=2 --master_port $MASTER_PORT run_lora.py \
        --deepspeed src/ft_chatglm_all/ds.config \
        --do_train \
        --train_file $data_path/$task/train.json \
        --cache_dir $data_path \
        --prompt_column input \
        --response_column target \
        --overwrite_cache \
        --model_name_or_path $glm_model_path \
        --output_dir $checkpoint_path/partition-$date/$task \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 196 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --max_steps $MAX_STEPS \
        --logging_steps 100 \
        --save_steps ${MAX_STEPS} \
        --learning_rate $LR \
        --lora_rank ${lora_rank} \
        --trainable ${lora_trainable} \
        --modules_to_save ${modules_to_save} \
        --lora_dropout ${lora_dropout} \
        --fp16

    deepspeed --num_gpus=2 --master_port $MASTER_PORT run_lora.py \
        --do_predict \
        --test_file $data_path/$task/test.json \
        --cache_dir $data_path \
        --prompt_column input \
        --response_column target \
        --overwrite_cache \
        --model_name_or_path $glm_model_path \
        --peft_path $checkpoint_path/partition-$date/$task/checkpoint-$MAX_STEPS \
        --output_dir submission/$task \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 196 \
        --per_device_eval_batch_size 4 \
        --predict_with_generate

done

task_list=("CMeIE" "CHIP-CTC")
MAX_STEPS=800
for task in ${task_list[@]}
do
    deepspeed --num_gpus=2 --master_port $MASTER_PORT run_lora.py \
        --deepspeed src/ft_chatglm_all/ds.config \
        --do_train \
        --train_file $data_path/$task/train.json \
        --cache_dir $data_path \
        --prompt_column input \
        --response_column target \
        --overwrite_cache \
        --model_name_or_path $glm_model_path \
        --output_dir $checkpoint_path/partition-$date/$task \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 196 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --max_steps $MAX_STEPS \
        --logging_steps 100 \
        --save_steps ${MAX_STEPS} \
        --learning_rate $LR \
        --lora_rank ${lora_rank} \
        --trainable ${lora_trainable} \
        --modules_to_save ${modules_to_save} \
        --lora_dropout ${lora_dropout} \
        --fp16

    deepspeed --num_gpus=2 --master_port $MASTER_PORT run_lora.py \
        --do_predict \
        --test_file $test_path/$task/test.json \
        --cache_dir $data_path \
        --prompt_column input \
        --response_column target \
        --overwrite_cache \
        --model_name_or_path $glm_model_path \
        --peft_path $checkpoint_path/partition-$date/$task/checkpoint-$MAX_STEPS \
        --output_dir submission/$task \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 196 \
        --per_device_eval_batch_size 4 \
        --predict_with_generate \
        --seed 3407
done

task_list=("CHIP-CDEE")
MAX_STEPS=500
for task in ${task_list[@]}
do
    deepspeed --num_gpus=2 --master_port $MASTER_PORT run_lora.py \
        --deepspeed src/ft_chatglm_all/ds.config \
        --do_train \
        --train_file $data_path/$task/train.json \
        --cache_dir $data_path \
        --prompt_column input \
        --response_column target \
        --overwrite_cache \
        --model_name_or_path $glm_model_path \
        --output_dir $checkpoint_path/partition-$date/$task \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 196 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --max_steps $MAX_STEPS \
        --logging_steps 100 \
        --save_steps ${MAX_STEPS} \
        --learning_rate $LR \
        --lora_rank ${lora_rank} \
        --trainable ${lora_trainable} \
        --modules_to_save ${modules_to_save} \
        --lora_dropout ${lora_dropout} \
        --fp16

    deepspeed --num_gpus=2 --master_port $MASTER_PORT run_lora.py \
        --do_predict \
        --test_file $test_path/$task/test.json \
        --cache_dir $data_path \
        --prompt_column input \
        --response_column target \
        --overwrite_cache \
        --model_name_or_path $glm_model_path \
        --peft_path $checkpoint_path/partition-$date/$task/checkpoint-$MAX_STEPS \
        --output_dir submission/$task \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 196 \
        --per_device_eval_batch_size 4 \
        --predict_with_generate \
        --seed 3407
done
