#!/bin/bash
#SBATCH -J openqwen_eval                       # Job name
#SBATCH -o logs/openqwen_eval.out                  # Name of stdout output log file (%j expands to jobID)
#SBATCH -e logs/openqwen_eval.err                  # Name of stderr output log file (%j expands to jobID)
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=8                       # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=512G
#SBATCH --qos=high_priority_nice
#SBATCH --partition=main
#SBATCH -t 72:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=8                       # Specify a list of generic consumable resources (per node)
##SBATCH --reservation=haoli_resv
#SBATCH --dependency=afterok:278467            # change this to automatically run after the finetune job

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

# Add mm_sequence_packing to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "PYTHONPATH is set to: $PYTHONPATH"

# add prismatic to Python path
export PYTHONPATH="$PROJECT_ROOT/prismatic-vlms/prismatic:$PYTHONPATH"
echo "PYTHONPATH is set to: $PYTHONPATH"

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
DATAPATH="/fsx/data/common"
CKPT_PATH="/fsx/users/haoli/datopenqwen/checkpoints/qwen2.5-1.5b-instruct-continue-training-llava_sft_4nodepretrained+stage-finetune+x7"
CKPTID="llava_sft_4nodepretrained"

# Print configuration
echo "Running with configuration:"
echo "  Data Path: $DATAPATH"
echo "  Checkpoint Path: $CKPT_PATH"
echo "  Checkpoint ID: $CKPTID"
echo "========================="

# Run the training script
./vlm-evaluation/eval.sh "$DATAPATH" "$CKPT_PATH" "$CKPTID"

echo "Job completed at $(date)"
