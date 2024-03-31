#!/encs/bin/tcsh

#SBATCH --job-name=gemma_training_job -A costa

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1 #On P6 cards this value MUST be: 1

#SBATCH --cpus-per-task=32

#SBATCH --ntasks-per-node=1

#SBATCH --mem=5G             ## Assign memory per node

if ( $?SLURM_CPUS_PER_TASK ) then

   setenv omp_threads $SLURM_CPUS_PER_TASK

else

   setenv omp_threads 1

endif

setenv OMP_NUM_THREADS $omp_threads



setenv RDZV_HOST `hostname -s`

setenv RDZV_PORT 29400

setenv endpoint ${RDZV_HOST}:${RDZV_PORT}

setenv CUDA_LAUNCH_BLOCKING 1

setenv NCCL_BLOCKING_WAIT 1

#setenv NCCL_DEBUG INFO

setenv NCCL_P2P_DISABLE 1

setenv NCCL_IB_DISABLE 1

source /speed-scratch/$USER/tmp/gemmaenv/bin/activate.csh #path where you have created your python venv

unsetenv CUDA_VISIBLE_DEVICES

# nproc_per_node=1 On P6 cards

srun torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$endpoint finetune_gemma_for_speed.py /speed-scratch/ /speed-scratch/$USER/genai/Results/ 

deactivate