#!/bin/bash

#SBATCH --account=def-XXX //update this accordingly....
#SBATCH --job-name=gemma_training_job
#SBATCH --nodes=1

#SBATCH --gpus-per-node=1 #On P6 cards this value MUST be: 1

#SBATCH --cpus-per-task=32

#SBATCH --ntasks-per-node=1

#SBATCH --mem=5G             ## Assign memory per node
#SBATCH --time=15:0:0


# set path to where you have the files. Update accordingly
cd /home/rraj/projects/def-XXX/rraj/GenAI_Project
source /home/rraj/projects/def-XXX/rraj/GenAI_Project/genAIEnv/bin/activate #path where you have created your python venv




# load any required modules
module spider python/3.9.6

#load pyarrow module
module load StdEnv/2023
module load arrow/15.0.1



# update this path accordingly
srun python  finetune_gemma.py --data_dir /home/rraj/projects/def-XXX/rraj/GenAI_Project/  


