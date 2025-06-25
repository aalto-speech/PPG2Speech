#!/usr/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-h200-18g-ia
#SBATCH --constraint=hopper
#SBATCH --gpus=1
#SBATCH --job-name="mfa_align"

module load mamba kaldi-strawberry sox
source activate mfa

corpus=$1
dictionary=$2
mfa_model=$3
output_dir=$4

mfa align --output_format json -d -j 8 -v --use_mp --clean --final_clean --overwrite \
    ${corpus} ${dictionary} ${mfa_model} ${output_dir}