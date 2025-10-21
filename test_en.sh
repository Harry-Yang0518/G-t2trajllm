#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

source /scratch/hy2611/miniconda/etc/profile.d/conda.sh
conda activate traj2llm


python test_T2TrajLLM_en.py \
  --texts  datasets/celebrity_biographies/texts.txt \
  --labels datasets/celebrity_biographies/labels.txt \
  --preds  datasets/celebrity_biographies/preds.txt \
  --run-inference  1 \
  --model gpt-4-turbo \
  --logdir logs/celebrity_biographies \
  --limit 0
