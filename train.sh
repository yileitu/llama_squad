#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32768
#SBATCH --gpus=a100-pcie-40gb:1

module load eth_proxy
module load gcc/9.3.0
module load cuda/12.1.1
conda activate subnet

python test_llama_squad.py \
--tokenizer_name=finetuned/final_merged_checkpoint \
--tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--num_samples 100


#python train_llama_squad.py \
#--model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
#--dataset_name data/squad_v2 \
#--fp16 \
#--max_seq_length 4096 \
#--per_device_train_batch_size 4 \
#--gradient_accumulation_steps 4 \
#--num_train_epochs 1 \
#--save_strategy no \
#--output_dir finetuned \
#--merge_and_push \
#--learning_rate=2e-7
