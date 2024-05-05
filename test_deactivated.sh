#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32768
#SBATCH --gpus=rtx_3090:1

module load eth_proxy
module load gcc/9.3.0
module load cuda/12.1.1
conda activate subnet

export LANG=zh
export MODEL_NAME=tinyllama-240k-503b

python test_llama_deactivation_new.py \
  --lang $LANG \
  --model_full_name $MODEL_NAME \
  --dataset data/mlqa/$LANG/dev \
  --output_csv_file results/mlqa/$LANG/$MODEL_NAME/primal.csv \
  --model_name finetuned/$MODEL_NAME/epoch8/final_merged_checkpoint \
  --tokenizer_name finetuned/$MODEL_NAME/epoch8/final_merged_checkpoint \
  --load_lang_neuron_position True \
  --corpus_name CC100 \
  --num_datapoints 5000 \
