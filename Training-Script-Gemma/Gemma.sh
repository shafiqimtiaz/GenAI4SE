#!/bin/bash
#SBATCH -J "T5-Exp1" 
#SBATCH --account=rraj
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rachna.raj@mail.concordia.ca

source /home/t_ereira/myenvs/bin/activate
conda activate /home/t_ereira/myenvs/envs/cerberus

python "/home/t_ereira/GenAI/TrainingScriptExp1.py" "/home/t_ereira/GenAI/train_preprocessed.csv" "/home/t_ereira/GenAI/test_preprocessed.csv" "/home/t_ereira/GenAI/Results/"