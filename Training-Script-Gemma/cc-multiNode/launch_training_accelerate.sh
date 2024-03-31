#!/bin/bash

source $SLURM_TMPDIR/ENV/bin/activate
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Node $SLURM_NODEID says: main node at $HEAD_NODE"
echo "Node $SLURM_NODEID says: Launching python script with accelerate..."

accelerate launch \
--multi_gpu \
--gpu_ids="all" \
--num_machines=$SLURM_NNODES \
--machine_rank=$SLURM_NODEID \
--num_processes=4 \ # This is the total number of GPUs across all nodes
--main_process_ip="$HEAD_NODE" \
--main_process_port=$HEAD_NODE_PORT \
srun python finetune_gemma.py --data_dir /home/rraj/projects/def-costa/rraj/GenAI_Project/