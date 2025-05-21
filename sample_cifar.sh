#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 48:00:00                  # Time limit (hh:mm:ss)
#SBATCH --constraint=h100
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption
#SBATCH -p kempner_h100
#SBATCH --account kempner_albergo_lab

DATASET_PATH="cifar10"
MODEL="mdlm"


# Setup environment
source setup_env.sh
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Expecting:
#  - MODEL (mdlm, udlm)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi

RUN_NAME="${MODEL}"
T=0
if [ "${MODEL}" = "mdlm" ]; then
  PARAMETERIZATION=subs
  DIFFUSION="absorbing_state"
  ZERO_RECON_LOSS=False
  time_conditioning=False
  sampling_use_cache=True
elif [ "${MODEL}" = "udlm" ]; then
  PARAMETERIZATION=d3pm
  DIFFUSION="uniform"
  ZERO_RECON_LOSS=True
  time_conditioning=True
  sampling_use_cache=False
else
  echo "MODEL must be one of mdlm, udlm"
  exit 1
fi

# To enable preemption re-loading, set `hydra.run.dir` or
srun python sample_cifar.py \
  is_vision=True \
  diffusion=${DIFFUSION} \
  parameterization=${PARAMETERIZATION} \
  T=${T} \
  time_conditioning=${time_conditioning} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data=cifar10 \
  data.train=${DATASET_PATH} \
  data.valid=${DATASET_PATH} \
  guidance=cfg-smc \
  guidance.condition=7 \
  guidance.gamma=3.0 \
  guidance.resample_fraction=0.125 \
  guidance.resample_threshold=0.1 \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=64 \
  backbone=unet \
  model=unet \
  optim.lr=2e-4 \
  eval.checkpoint_path="${PWD}/outputs/cifar10/mdlm/checkpoints/best.ckpt" \
  eval.generated_samples_path="${PWD}/outputs/cifar10/${RUN_NAME}/samples-smc-fid" \
  lr_scheduler=constant_warmup \
  lr_scheduler.num_warmup_steps=5000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
  trainer.max_steps=1_000_000 \
  trainer.val_check_interval=10_000 \
  +trainer.check_val_every_n_epoch=null \
  training.guidance.cond_dropout=0.1 \
  eval.generate_samples=True \
  sampling.num_sample_batches=3125 \
  sampling.batch_size=16 \
  sampling.use_cache=${sampling_use_cache} \
  sampling.steps=128 \
  wandb.name="cifar10_${RUN_NAME}_large" \
  hydra.run.dir="${PWD}/outputs/cifar10/${RUN_NAME}" 
