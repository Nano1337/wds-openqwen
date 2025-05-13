#!/bin/bash
set -euxo pipefail

# Usage: ./train.sh [CKPTID] [GLOBAL_BATCH_SIZE] [PER_GPU_BATCH_SIZE] [DATASET_PATH]
# Example:
#   ./train.sh exp1 64 8 s3://your-bucket/multimodal-data/{00000..00099}.tar  # Stream from S3
#   ./train.sh exp1 64 8 s3://your-bucket/multimodal-data/shards.txt          # Using shard list file
#   ./train.sh exp1 64 8 /path/to/local/raw_webdataset                        # Local WebDataset files

CKPTID=$1
BSZ=$2
PER_GPU_BSZ=$3
DATASET_PATH=${4:-"Open-Qwen2VL-Data"}  # Default dataset path if not provided

# Echo Python path for debugging
echo "Using PYTHONPATH: $PYTHONPATH"

# Dynamic packing configuration
WORKERS=16
SHUFFLE_BUFFER=5000
SAMPLES_PER_PACK=256
echo "Using dynamic sequence packing with $WORKERS workers"

# Dynamic packing specific args
DYNAMIC_ARGS="--dataset.workers $WORKERS --dataset.shuffle_buffer $SHUFFLE_BUFFER --dataset.samples_per_pack $SAMPLES_PER_PACK"

# Ensure we include the current directory and any parent directories in the python path for imports
if [ -z "$PYTHONPATH" ]; then
  export PYTHONPATH="$(pwd)"
else
  export PYTHONPATH="$(pwd):$PYTHONPATH"
fi
echo "Updated PYTHONPATH: $PYTHONPATH"

# detect multi-node settings (defaults to 1 node / rank 0)
NNODES=${SLURM_JOB_NUM_NODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
# master = first hostname in the allocated list
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}
# Diagnostic logging for distributed training
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Distributed config: nnodes=$NNODES, nproc_per_node=8, node_rank=$NODE_RANK, master_addr=$MASTER_ADDR, master_port=$MASTER_PORT, rdzv_id=$SLURM_JOB_ID, rdzv_backend=c10d"

torchrun \
  --nnodes $NNODES \
  --nproc_per_node 8 \
  --node_rank $NODE_RANK \
  --rdzv_id $SLURM_JOB_ID \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  prismatic-vlms/scripts/pretrain.py \
  --stage datology-pretrain \
  --model.type "one-stage+7b" \
  --model.model_id qwen2.5-1.5b-instruct-continue-training-${CKPTID} \
  --model.arch_specifier "no-align+avgpool" \
  --model.vision_backbone_id "siglip-vit-so400m-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "qwen2.5-1.5b-instruct" \
  --model.pretrain_global_batch_size ${BSZ} \
  --model.pretrain_per_device_batch_size ${PER_GPU_BSZ} \
  --model.pretrain_epochs 1 \
  --model.pretrain_train_strategy "fsdp-full-shard" \
  --model.pretrain_max_steps 3748 \
  --mount_path Qwen \
  --run_root_dir checkpoints/ \
  --dataset.type "pretrain" \
  --dataset.dataset_root_dir ${DATASET_PATH} \
  ${DYNAMIC_ARGS}

echo "Training completed at $(date)"


