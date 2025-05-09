#!/bin/bash
#SBATCH -J s3_parallel_download
#SBATCH -o logs/s3_parallel_download.out
#SBATCH -e logs/s3_parallel_download.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=512G
#SBATCH --qos=high_priority_nice
#SBATCH --partition=main
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=0
#SBATCH --reservation=haoli_resv

set -euo pipefail

# ==== Step 1: Validate input ====
if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch s3_parallel_download.sbatch <s3://bucket/prefix/>"
  exit 1
fi

S3_DIR="$1"
TARGET_DIR="/fsx/data/common/MAmmoTH-VL-Instruct-12M/single_image_data"

# ==== Step 2: Extract bucket and prefix path ====
BUCKET=$(echo "$S3_DIR" | sed -E 's|s3://([^/]*).*|\1|')
PREFIX=$(echo "$S3_DIR" | sed -E 's|s3://[^/]*/(.*)|\1|')

echo "🪣 Bucket: $BUCKET"
echo "📂 Prefix: $PREFIX"
echo "📥 Downloading to: $TARGET_DIR"

# ==== Step 3: Prepare directory ====
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# ==== Step 4: List all keys ====
aws s3 ls "s3://$BUCKET/$PREFIX" --recursive | awk '{print $4}' > s3_files.txt

echo "📄 Number of files to download: $(wc -l < s3_files.txt)"

# ==== Step 5: Download in parallel ====
cat s3_files.txt | xargs -I {} -P $SLURM_CPUS_PER_TASK aws s3 cp "s3://$BUCKET/{}" "{}"

echo "✅ Parallel S3 download complete."