# Datology Open-Qwen2VL

This is the internal port of the Open-Qwen2VL repo. 

## Setup

Please make sure you have `uv` installed for dependency management. 

Then simply run: 
```Shell
uv sync
uv pip install -e vlm-evaluation
uv pip install -e prismatic-vlms
```

## Training

### Train Data Prep

After downloading the data, run multimodal sequence packing: 

```bash

bash mm_sequence_packing/multiprocess_sequence_packing_image_to_pil.sh 0 4 504 datacomp /home/ec2-user/Open-Qwen2VL-Data 
bash mm_sequence_packing/multiprocess_sequence_packing_image_to_pil.sh 0 4 326 ccs /home/ec2-user/Open-Qwen2VL-Data

```


## Evals

### Eval Data Prep

(TODO) talk about huggingface token setup

To set up the data (if it doesn't already exist in `/fsx/data/common/vlm_eval_data`), run:
```Shell
python data_filtering/prepare_data.py
```
