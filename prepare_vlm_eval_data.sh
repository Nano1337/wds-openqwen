#!/bin/bash
# prepare_vlm_eval_data.sh
# Script to prepare VLM evaluation datasets

# Set the root directory for data
ROOT_DIR="/fsx/data/common/vlm_eval_data"
SHOTS=0  # Number of few-shot examples (0 for zero-shot)

# Set HF token environment variable using the existing token file
export HF_TOKEN="evals"

# Define datasets to prepare
DATASETS=(
    "ai2d"
    "text-vqa"
    "pope"
    "mmmu"
    "mmbench"
    "seedbench"
    "mmstar"
    "mathvista"
)

# Function to prepare a dataset
prepare_dataset() {
    local dataset=$1
    
    echo "=== Preparing dataset: ${dataset} ==="
    
    # Clear existing dataset files if needed
    if [ -d "${ROOT_DIR}/datasets/${dataset}" ]; then
        echo "Removing existing files for ${dataset}..."
        rm -rf "${ROOT_DIR}/datasets/${dataset}/*"
    fi
    
    # Run the preparation script
    python vlm-evaluation/scripts/datasets/prepare.py \
        --dataset_family ${dataset} \
        --root_dir ${ROOT_DIR} \
        --shots ${SHOTS} \
        --hf_token HF_TOKEN
    
    echo "=== Completed preparation for ${dataset} ==="
    echo ""
}

# Main execution
echo "Starting VLM evaluation data preparation"
echo "Root directory: ${ROOT_DIR}"
echo "-------------------------------------"

# Create root directory if it doesn't exist
mkdir -p ${ROOT_DIR}

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    prepare_dataset "$dataset"
done

echo "-------------------------------------"
echo "All datasets prepared successfully!"
echo "Data is available at: ${ROOT_DIR}"
