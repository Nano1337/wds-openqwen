#!/bin/bash
#SBATCH -J untar_batched_pigz
#SBATCH -o logs/untar_batched_pigz.out
#SBATCH -e logs/untar_batched_pigz.err
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

# ==== Step 1: Check input ====
if [ "$#" -ne 2 ]; then
  echo "Usage: sbatch untar_batched_pigz.sbatch <source_dir_with_tars> <target_extract_dir>"
  exit 1
fi

SOURCE_DIR="$1"
TARGET_DIR="$2"

echo "📦 Source dir: $SOURCE_DIR"
echo "📁 Target dir: $TARGET_DIR"

mkdir -p "$TARGET_DIR"

cd "$SOURCE_DIR"

# ==== Step 2: Find all .tar.gz files ====
find . -maxdepth 1 -type f -name '*.tar.gz' | sort > tar_list.txt
NUM_FILES=$(wc -l < tar_list.txt)

echo "🧾 Found $NUM_FILES .tar.gz files. Processing in batches of 25..."

# ==== Step 3: Batch in groups of 25 ====
split -l 5 tar_list.txt tar_batch_

for BATCH in tar_batch_*; do
  echo "🚀 Starting batch: $BATCH"
  
  cat "$BATCH" | xargs -I {} -P 5 bash -c '
    echo "🗃️ Extracting {} with pigz (7 threads)..."
    if tar --use-compress-program="pigz -d -p 30" -xf "{}" -C "'"$TARGET_DIR"'"; then
      echo "🧹 Deleting {}"
      rm "{}"
    else
      echo "❌ Failed to extract {}"
    fi
  '

  echo "✅ Batch $BATCH complete."
done

echo "🎉 All batches completed successfully."