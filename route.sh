 
dataset=spider
flags=1_1_1_1 # 0_0_0_0 for TS, 1_1_0_0 for SL,  0_0_1_0 for NC, 0_0_0_1 for CW

for dataset in spider bird
do
for flags in 0_0_0_0 1_1_1_1
do

data_path=your data_path
mode=dev
output_path=your output_path
gpus=1

batch_size=32
LLM_model=your sft model path
tag=llama3_8b_MSFT  # if online in tag, customize your online LLM
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

batch_size=32
LLM_model=your instruct model path
tag=llama3_8b_instruct
# for LLM w/o sft/msft
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
LLM_model=qwen2.5-14b-msft
tag=qwen2.5_14b_MSFT  # if online in tag, customize your online LLM
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

