#!/bin/bash

# 定义参数的取值范围
declare -a LANGS=(ar de en es hi vi zh)
declare -a MODELS=(tinyllama-1431k-3T tinyllama-240k-503b tinyllama-480k-1T tinyllama-50k-105b)
#declare -a LANGS=(ar)
#declare -a MODELS=(tinyllama-1431k-3T)

# 遍历所有可能的参数组合
for language in "${LANGS[@]}"; do
  for model in "${MODELS[@]}"; do
    for load_lang_neuron in True False; do
      if [ "$load_lang_neuron" = "True" ]; then
        OUTPUT_PATH="results/mlqa/$language/$model/deactivated"
      else
        OUTPUT_PATH="results/mlqa/$language/$model/primal"
      fi
      export OUTPUT_PATH
    sbatch <<EOT
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


python test_llama_deactivation_new.py \
  --lang $language \
  --model_full_name $model \
  --dataset data/mlqa \
  --output_csv_path $OUTPUT_PATH \
  --model_name finetuned/$model/epoch8/final_merged_checkpoint \
  --tokenizer_name finetuned/$model/epoch8/final_merged_checkpoint \
  --load_lang_neuron_position $load_lang_neuron \
  --corpus_name CC100 \
  --num_datapoints 5000
EOT
    done
  done
done
