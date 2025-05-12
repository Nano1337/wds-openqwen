#!/bin/bash
#SBATCH -J openqwen_train                       # Job name
#SBATCH -o logs/openqwen_train.out                  # Name of stdout output log file (%j expands to jobID)
#SBATCH -e logs/openqwen_train.err                  # Name of stderr output log file (%j expands to jobID)
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=8                       # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=512G
#SBATCH --qos=high_priority_nice
#SBATCH --partition=main
#SBATCH -t 72:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=8                       # Specify a list of generic consumable resources (per node)
#SBATCH --reservation=haoli_resv2
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

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

# Add NCCL timeout and retry settings to prevent communication hang
export NCCL_SOCKET_NTHREADS=8              # More threads for socket communications
export NCCL_NSOCKS_PERTHREAD=4             # More sockets per thread
export NCCL_SOCKET_IFNAME=^lo,docker       # Avoid certain interfaces
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # New version of NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_BLOCKING_WAIT=0          # New version of NCCL_BLOCKING_WAIT

# Use longer timeout values (30 minutes)
export NCCL_TIMEOUT=1800000                # NCCL timeout in ms (30 minutes)
export TORCH_DISTRIBUTED_TIMEOUT=1800      # PyTorch distributed timeout in seconds (30 minutes)
export NCCL_MIN_NCHANNELS=8                # Use more channels for better stability

# AWS S3 connection settings
export AWS_REQUEST_PAYER=requester          # For shared buckets
export AWS_MAX_ATTEMPTS=20                  # Retry up to 20 times on S3 operations
export AWS_RETRY_MODE=adaptive              # Adaptive backoff 
export S3_CONNECT_TIMEOUT=60                # 60 second connection timeout
export S3_READ_TIMEOUT=300                  # 5 minute read timeout
export HTTP_TIMEOUT=300                     # HTTP timeout for WebDataset fetches

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
GLOBAL_BSZ=256                                               # Global batch size
PER_GPU_BSZ=4                                               # Per GPU batch size

# Hardcoded dataset paths with WebDataset :: operator to concatenate
DATASET_PATH="s3://datology-research/haoli/Open-Qwen2VL-Data/ccs_webdataset/{00000..01302}.tar::s3://datology-research/haoli/Open-Qwen2VL-Data/datacomp_medium_mlm_filter_su_85_union_dfn_webdataset/{00000000..00002012}.tar"

# Print configuration
echo "Running with configuration:"
echo "  Checkpoint ID: $CKPTID"
echo "  Global batch size: $GLOBAL_BSZ"
echo "  Per GPU batch size: $PER_GPU_BSZ"
echo "  Dataset path: $DATASET_PATH"
echo "========================="

# Run the training script
./prismatic-vlms/train.sh "$CKPTID" "$GLOBAL_BSZ" "$PER_GPU_BSZ" "$DATASET_PATH"

echo "Job completed at $(date)"
