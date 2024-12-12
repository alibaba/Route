 
dataset=spider
flags=1_1_1_1 # 0_0_0_0 for TS, 1_1_0_0 for SL,  0_0_1_0 for NC, 0_0_0_1 for CW

for dataset in spider bird
do
for flags in 0_0_0_0 1_1_1_1
do

data_path=/disk2/qinyang/SFT/LLama_factory_qy/A_qy/data/dataset
mode=dev
output_path=/disk2/qinyang/SFT/LLama_factory_qy/ROUTE-SQL/output_new/
gpus=1

batch_size=16
# /disk1/chenchao/Models/nl2sql-qwen2.5-14b-sft
LLM_model=/disk2/qinyang/SFT/LLama_factory_qy/pre_model/rebuttal/full_Llama-3-8B-Instruct_MSFT_clean_1W_epoch2_1e-5_4x16
tag=llama3_8b_MSFT_clean  # if online in tag, customize your online LLM
# for sft/msft LLM 
CUDA_VISIBLE_DEVICES=0 python route.py --data_path $data_path \
    --dataset $dataset \
    --output_path $output_path \
    --mode $mode \
    --LLM_model $LLM_model \
    --batch_size $batch_size\
    --tag $tag \
    --gpus $gpus\
    --flags $flags \
    --eval_sft 1\ 


batch_size=16
LLM_model=/disk2/qinyang/weights/LLama3
tag=llama3_8b
# for sft/msft LLM
CUDA_VISIBLE_DEVICES=0 python route.py --data_path $data_path \
    --dataset $dataset \
    --output_path $output_path \
    --mode $mode \
    --LLM_model $LLM_model \
    --batch_size $batch_size\
    --tag $tag \
    --gpus $gpus\
    --flags $flags \
    --eval_sft 0\ 



batch_size=16
LLM_model=/disk1/chenchao/Models/nl2sql-qwen2.5-14b-sft
tag=qwen2.5_14b_MSFT_clean  # if online in tag, customize your online LLM
# for sft/msft LLM 
CUDA_VISIBLE_DEVICES=0 python route.py --data_path $data_path \
    --dataset $dataset \
    --output_path $output_path \
    --mode $mode \
    --LLM_model $LLM_model \
    --batch_size $batch_size\
    --tag $tag \
    --gpus $gpus\
    --flags $flags \
    --eval_sft 1\ 
    
done
done

