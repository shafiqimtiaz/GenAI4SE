#!/bin/bash

#SBATCH --account=def-costa
#SBATCH --job-name=gemma_training_job
#SBATCH --nodes=1

#SBATCH --gpus-per-node=1 #On P6 cards this value MUST be: 1

#SBATCH --cpus-per-task=32

#SBATCH --ntasks-per-node=1

#SBATCH --mem=20000M             ## Assign memory per node
#SBATCH --time=4-00:0



cd /home/rraj/projects/def-costa/rraj/genai/singleNode
source /home/rraj/projects/def-costa/rraj/genai/genAIEnv/bin/activate #path where you have created your python venv




# load any required modules
module spider python/3.9.6
module load cuda
module load llvm


#load pyarrow module
module load StdEnv/2023
module load arrow/15.0.1




srun python  finetune_gemma.py --data_dir /home/rraj/projects/def-costa/rraj/genai/ evaluation.gpu_collect=True  


