from pathlib import Path
from prismatic.preprocessing.download import convert_to_jpg
import sys
import os
sys.path.append('/fsx/users/haoli/datopenqwen/mm_sequence_packing')

def preprocess_ocrvqa(root_dir: str) -> None:
    convert_to_jpg(Path(os.path.join(root_dir, "ocr_vqa", "images")))


if __name__ == "__main__":
    preprocess_ocrvqa("/fsx/data/common/llava-1.5-665k-instructions/train_split")
