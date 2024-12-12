# https://github.com/hiyouga/LLaMA-Factory

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ 

NPROC_PER_NODE=8
NODE_RANK=0  
MASTER_ADDR=localhost
MASTER_PORT=29502
NNODES=1 

## customize
DS_CONFIG_PATH=examples/deepspeed/ds_z3_config.json
DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# for llama3 8B
per_device_train_batch_size=4
gradient_accumulation_steps=16
learning_rate=1e-5
num_train_epochs=2

model=Llama-3-8B-Instruct
MODEL_PATH=./models/Llama-3-8B-Instruct

finetuning_type=full 
dataset=SFT_data,NC_clean_1W,CL_clean_1W
run_name=MSFT
tag=${model}_${run_name}_epoch${num_train_epochs}_${learning_rate}_${per_device_train_batch_size}x${gradient_accumulation_steps}
OUTPUT_PATH=./outputs/${finetuning_type}_${tag} 
mkdir -p ${OUTPUT_PATH}

torchrun $DISTRIBUTED_ARGS src/train.py --deepspeed $DS_CONFIG_PATH --stage sft --do_train --use_fast_tokenizer --flash_attn auto --run_name $run_name --model_name_or_path $MODEL_PATH --dataset $dataset --template llama3 --finetuning_type $finetuning_type --lora_target q_proj,v_proj  --output_dir $OUTPUT_PATH --overwrite_cache --overwrite_output_dir --warmup_ratio 0.05 --weight_decay 0.1 --per_device_train_batch_size $per_device_train_batch_size --gradient_accumulation_steps $gradient_accumulation_steps --ddp_timeout 9000 --learning_rate $learning_rate --preprocessing_num_workers=16 --lr_scheduler_type cosine --logging_steps 1 --cutoff_len 4096 --save_steps 1000000000000 --plot_loss --report_to none --num_train_epochs $num_train_epochs --bf16 | tee ${OUTPUT_PATH}/log.txt


# for qwen2.5 7B
per_device_train_batch_size=4
gradient_accumulation_steps=16
learning_rate=1e-5
num_train_epochs=2

model=Qwen-2.5-7B-Instruct
MODEL_PATH=./models/Qwen-2.5-7B-Instruct

finetuning_type=full 
dataset=SFT_data,NC_clean_1W,CL_clean_1W
run_name=MSFT
tag=${model}_${run_name}_epoch${num_train_epochs}_${learning_rate}_${per_device_train_batch_size}x${gradient_accumulation_steps}
OUTPUT_PATH=./outputs/${finetuning_type}_${tag} 
mkdir -p ${OUTPUT_PATH}

torchrun $DISTRIBUTED_ARGS src/train.py --deepspeed $DS_CONFIG_PATH --stage sft --do_train --use_fast_tokenizer --flash_attn auto --run_name $run_name --model_name_or_path $MODEL_PATH --dataset $dataset --template qwen2 --finetuning_type $finetuning_type --lora_target q_proj,v_proj  --output_dir $OUTPUT_PATH --overwrite_cache --overwrite_output_dir --warmup_ratio 0.05 --weight_decay 0.1 --per_device_train_batch_size $per_device_train_batch_size --gradient_accumulation_steps $gradient_accumulation_steps --ddp_timeout 9000 --learning_rate $learning_rate --preprocessing_num_workers=16 --lr_scheduler_type cosine --logging_steps 1 --cutoff_len 4096 --save_steps 1000000000000 --plot_loss --report_to none --num_train_epochs $num_train_epochs --bf16 | tee ${OUTPUT_PATH}/log.txt


# for Qwen2.5 14B
per_device_train_batch_size=2
gradient_accumulation_steps=16
learning_rate=1e-5
num_train_epochs=2

model=Qwen-2.5-14B-Instruct
MODEL_PATH=./models/Qwen-2.5-7B-Instruct

finetuning_type=full 
dataset=SFT_data,NC_clean_1W,CL_clean_1W
run_name=MSFT
tag=${model}_${run_name}_epoch${num_train_epochs}_${learning_rate}_${per_device_train_batch_size}x${gradient_accumulation_steps}
OUTPUT_PATH=./outputs/${finetuning_type}_${tag} 
mkdir -p ${OUTPUT_PATH}

torchrun $DISTRIBUTED_ARGS src/train.py --deepspeed $DS_CONFIG_PATH --stage sft --do_train --use_fast_tokenizer --flash_attn auto --run_name $run_name --model_name_or_path $MODEL_PATH --dataset $dataset --template qwen2 --finetuning_type $finetuning_type --lora_target q_proj,v_proj  --output_dir $OUTPUT_PATH --overwrite_cache --overwrite_output_dir --warmup_ratio 0.05 --weight_decay 0.1 --per_device_train_batch_size $per_device_train_batch_size --gradient_accumulation_steps $gradient_accumulation_steps --ddp_timeout 9000 --learning_rate $learning_rate --preprocessing_num_workers=16 --lr_scheduler_type cosine --logging_steps 1 --cutoff_len 4096 --save_steps 1000000000000 --plot_loss --report_to none --num_train_epochs $num_train_epochs --bf16 | tee ${OUTPUT_PATH}/log.txt

