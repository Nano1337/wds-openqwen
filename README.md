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

Stage 1:
- Refactored code so it does multimodal sequence packing on the fly. 
- Offline packing takes up way too much disk storage

Stage 2:
- Need to download Mammoth SFT dataset as instructed in `docs/README_ORIGINAL.md`. 
- Only need the `single_image_data` folder and the `mammoth_si_10M.json` file for data prep. 
- The paper complains that they don't have 200GB RAM to load `mammoth_si_10M.json` into memory during training so they write out 10M json files and use `mammoth_si_10M_simple.json` as a pointer file during training. This is created by running `data_prepare/split_mammoth_10m.py`.




## Evals

### Eval Data Prep

(TODO) talk about huggingface token setup

To set up the data (if it doesn't already exist in `/fsx/data/common/vlm_eval_data`), run:
```Shell
./prepare_vlm_eval_data.sh
```
