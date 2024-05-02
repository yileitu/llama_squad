#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32768
#SBATCH --gpus=v100:1

module load eth_proxy
module load gcc/9.3.0
module load cuda/12.1.1
conda activate subnet


python test_llama_deactivation_new.py \
  --dataset data/mlqa/toy \
  --output_csv_file results/mlqa/toy/primal_new.csv \
  --model_name finetuned/final_merged_checkpoint \
  --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --num_samples 100 \
  --load_lang_neuron_position False \
  --corpus_name CC100 \
  --num_datapoints 5000 \
  --model_full_name tinyllama-1431k-3T \
