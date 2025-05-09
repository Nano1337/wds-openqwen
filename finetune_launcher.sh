#!/bin/bash
#SBATCH -J openqwen_finetune                       # Job name
#SBATCH -o logs/openqwen_finetune.out                  # Name of stdout output log file (%j expands to jobID)
#SBATCH -e logs/openqwen_finetune.err                  # Name of stderr output log file (%j expands to jobID)
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=8                       # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=512G
#SBATCH --qos=high_priority_nice
#SBATCH --partition=main
#SBATCH -t 72:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=8                       # Specify a list of generic consumable resources (per node)
#SBATCH --reservation=haoli_resv
########
# Manually set and enter project root (FSX mount)
export PROJECT_ROOT="/fsx/users/haoli/datopenqwen"
cd "$PROJECT_ROOT/"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Activate virtualenv
source "$PROJECT_ROOT/.venv/bin/activate"

# Point to venv python
export PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

# Log some information
echo "==== Job Information ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================="

########
# Set training configuration
CKPTID="qwen_vlm_dp"                                         # Checkpoint ID 
CKPT_PATH="/fsx/users/haoli/datopenqwen/checkpoints/pretrain+qwen2.5-1.5b-instruct-continue-training-qwen_vlm_dp+stage-dynamic-pretrain+x7"                                               # Checkpoint Path
DATA_ROOT="/fsx/data/common/MAmmoTH-VL-Instruct-12M"                                               # Data Root

# Print configuration
echo "Running with configuration:"
echo "  Checkpoint ID: $CKPTID"
echo "  Checkpoint Path: $CKPT_PATH"
echo "  Data Root: $DATA_ROOT"
echo "========================="

# Run the training script
./prismatic-vlms/fine_tune_mammoth.sh "$CKPT_PATH" "$CKPTID" "$DATA_ROOT" 

echo "Job completed at $(date)"
