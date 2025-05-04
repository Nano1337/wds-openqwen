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

## Evals

### Eval Data Prep

(TODO) talk about huggingface token setup

To set up the data (if it doesn't already exist in `/fsx/data/common/vlm_eval_data`), run:
```Shell
python data_filtering/prepare_data.py
```
