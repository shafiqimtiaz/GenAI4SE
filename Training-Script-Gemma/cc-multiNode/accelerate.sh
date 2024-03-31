#!/bin/bash
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4 
#SBATCH --mem=16000M       
#SBATCH --output=%N-%j.out

## Create a virtualenv and install accelerate + its dependencies on all nodes ##
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env.sh

export HEAD_NODE=$(hostname) # store head node's address
export HEAD_NODE_PORT=34567 # choose a port on the main node to start accelerate's main process

srun launch_training_accelerate.sh