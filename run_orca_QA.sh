#!/bin/bash -l
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --mem=30G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu 10
#SBATCH --constraint=[v100]

echo "Loading anaconda..."

conda activate ./env

echo "...Anaconda env loaded"

mkdir -p gpu_logs
mkdir -p results

# a background job that logs GPU utilization to a file every second

nvidia-smi --query-gpu=index,timestamp,name,utilization.gpu,utilization.memory,pstate,memory.total,memory.free,memory.used --format=csv -l 1 > gpu_logs/gpu_utilization_$SLURM_JOB_ID.log &
GPU_LOG_PID=$!

#model_name_or_path=core42/jais-13b
model_name_or_path=hf-ep3-ba655000 # running
#model_name_or_path=AceGPT-7B
#model_name_or_path=converted_15
#model_name_or_path=FreedomIntelligence/AceGPT-7B
#model_name_or_path=HF-ALM
#config_name=hf-ep3-ba655000
#config_name=AceGPT-7B

task="qa"
seed=41
cache_dir="./cache"
tokenizer_name=None
model_revision='main'
use_auth_token=False
num_train_epochs=10
max_seq_len=512
bs=48 # batch_size
fp16="False"
trust_remote_code="False"
if_accelerate="False" 
freeze_layers="False"
return_dict=False

if [[ "$model_name_or_path" == "hf-ep3-ba655000" ]]; then
#if [[ "$model_name_or_path" == "AceGPT-7B" ]]; then
    trust_remote_code="False"
    bs=8
elif [[ "$model_name_or_path" == "core42/jais-13b" ]]; then
    fp16="True"
    trust_remote_code="True"
    if_accelerate="True" 
    bs=8
    freeze_layers="True" 
elif [[ "$model_name_or_path" == "FreedomIntelligence/AceGPT-13B" ]] || [[ "$model_name_or_path" == "FreedomIntelligence/AceGPT-7B" ]]; then
    fp16="True"
    trust_remote_code="True"
    if_accelerate="False" 
    bs=8
    freeze_layers="True" 
else
    fp16="False"
    trust_remote_code="True"
    bs=16
fi


if [ "$if_accelerate" == "True" ]; then
    echo "Launching with accelerator..."
    
    accelerate launch --config_file cache/default_config.yaml run_orca_qa.py --model_name_or_path  $model_name_or_path \
                          --dataset_name ./data \
                          --dataset_config_name $task \
                          --cache_dir $cache_dir \
                          --freeze_layers $freeze_layers \
                          --do_eval \
                          --per_device_train_batch_size $bs\
                          --per_device_eval_batch_size $bs\
                          --trust_remote_code $trust_remote_code \
                          --learning_rate 3e-5 \
                          --num_train_epochs $num_train_epochs \
                          --max_seq_length $max_seq_len \
                          --doc_stride 128 \
                          --fp16 $fp16 \
                          --output_dir ./output \
                          --evaluation_strategy epoch \
                          --logging_strategy epoch \
                          --save_strategy no \
                          --seed $seed \
                          --overwrite_output_dir
else
    echo "Launching without accelerator..."
    python run_orca_qa.py --model_name_or_path  $model_name_or_path \
                      --dataset_name ./data \
                      --dataset_config_name $task \
                      --cache_dir $cache_dir \
                      --freeze_layers $freeze_layers \
                      --do_eval \
                      --do_predict \
                      --per_device_train_batch_size $bs\
                      --per_device_eval_batch_size $bs\
                      --trust_remote_code $trust_remote_code \
                      --learning_rate 3e-5 \
                      --num_train_epochs $num_train_epochs \
                      --max_seq_length $max_seq_len \
                      --doc_stride 128 \
                      --fp16 $fp16 \
                      --output_dir ./output \
                      --evaluation_strategy epoch \
                      --logging_strategy epoch \
                      --save_strategy no \
                      --seed $seed \
                      --overwrite_output_dir
fi
kill $GPU_LOG_PID
