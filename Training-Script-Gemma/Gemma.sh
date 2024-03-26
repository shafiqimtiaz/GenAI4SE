#!/bin/bash
#SBATCH -J "T5-Exp1" 
#SBATCH --account=rraj
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rachna.raj@mail.concordia.ca

source /home/t_ereira/myenvs/bin/activate
conda activate /home/t_ereira/myenvs/envs/cerberus

python "/home/rraj/GenAI/TrainingScriptExp1.py" "/home/rraj/GenAI/train_preprocessed.csv" "/home/rraj/GenAI/test_preprocessed.csv" "/home/rraj/GenAI/Results/"