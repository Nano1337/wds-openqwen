from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER
from llava.mm_utils import tokenizer_image_token
import uuid
import pickle
import shutil
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
from pathlib import Path
import copy

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from transformers import AutoTokenizer
from transformers import SiglipImageProcessor

import pandas as pd
import io
import base64

from tqdm import tqdm
from typing import List, Tuple
import logging
import os, sys
import torch
import webdataset as wds
import json
import functools

END_CUNCK_TOKEN = "<|endofchunk|>"
NUM_IMAGES_TOKEN = 144

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("evaluation test")


def truncate_end_image_token(input_tensor):
    reversed_tensor = input_tensor.flip(dims=[0])

    # Find the first index that is not IMAGE_TOKEN_INDEX
    first_non_minus_100_index = (reversed_tensor != IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0]

    # Slice the tensor up to that index and reverse it back
    return reversed_tensor[first_non_minus_100_index:].flip(dims=[0])


def preprocess_obelics_interleaved(
    sample,
    tokenizer,
    image_processor,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """
    # image transform includes the padding process and normalization process
    
    info = json.loads(sample[1])
    info['key'] = sample[0]

    texts = info["texts"]

    # load images first to find which ones are valid
    image_tensors, idxs = [], []
    for i, sample_image in enumerate(info["image_info"]):
        if not sample_image or "image_base64" not in sample_image:
            idxs.append(i)
            continue
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        # filter to images >= 10KB
        # if len(rawbytes) // 1000 <= MIN_KB:
        #     continue

        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        if image.size[0] == 1 or image.size[1] == 1:
            continue
        
        # no need to resize
        original_width, original_height = image.size
        if min(original_width, original_height) > 384:
            if original_width > original_height:
                new_height = 384
                new_width = int(384 / original_width * original_width)
            else:
                new_width = 384
                new_height = int(384 / original_height * original_height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    

        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # image_tensors.append(image_tensor)
        image_tensors.append(image)
        idxs.append(i)

    if len(image_tensors) == 0 or len(image_tensors) >= 10:
        return None

    # preprocess and pad images
    texts = [texts[idx] for idx in idxs]
    texts =  [DEFAULT_IMAGE_TOKEN+"\n" if x is None else x.replace(DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER)+END_CUNCK_TOKEN for x in texts]
    corpus = "".join(texts)

    input_ids = tokenizer_image_token(corpus, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    if input_ids[0] != tokenizer.bos_token_id:
        input_ids = torch.cat([torch.LongTensor([tokenizer.bos_token_id]), input_ids])
    
    if input_ids.shape[-1] > 6000:
        return None

    assert len(image_tensors) == ((input_ids == IMAGE_TOKEN_INDEX).sum())

    # Some examples contain many images at the end, which will not be attended on due to auto-regressive learning, truncate them
    if input_ids[-1] == IMAGE_TOKEN_INDEX:
        input_ids = truncate_end_image_token(input_ids)
    
    # recompute the num of images
    final_num_images = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
    
    # if all the images are at the end, then the doc becomes text-only doc, discard it
    if final_num_images == 0:
        return None
    
    assert len(image_tensors[:final_num_images]) == final_num_images
    
    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids = torch.cat([input_ids, torch.LongTensor([tokenizer.eos_token_id])])
    
    # assert bos token
    assert input_ids[0] == tokenizer.bos_token_id

    info['length'] = (NUM_IMAGES_TOKEN - 1) * len(image_tensors) + input_ids.shape[-1]

    return (image_tensors[:final_num_images], input_ids, info)


def preprocess_caption(
    sample,
    tokenizer,
    image_processor,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """
    # image transform includes the padding process and normalization process

    image, text, info = sample

    # load images first to find which ones are valid
    if image.size[0] == 1 or image.size[1] == 1:
        return None
    
    # image resize if bigger than max size
    original_width, original_height = image.size
    if min(original_width, original_height) > 384:
        if original_width > original_height:
            new_height = 384
            new_width = int(384 / original_width * original_width)
        else:
            new_width = 384
            new_height = int(384 / original_height * original_height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # image_tensors = torch.zeros((1, 3, 384, 384), dtype=torch.float32)
    # else:
        # image_tensors = image_processor.preprocess(image, return_tensors='pt')['pixel_values']

    text = DEFAULT_IMAGE_TOKEN+"\n" + text.replace(DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER)

    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    if input_ids[0] != tokenizer.bos_token_id:
        input_ids = torch.cat([torch.LongTensor([tokenizer.bos_token_id]), input_ids])

    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids = torch.cat([input_ids, torch.LongTensor([tokenizer.eos_token_id])])
    
    # assert bos token
    assert input_ids[0] == tokenizer.bos_token_id

    info['length'] = NUM_IMAGES_TOKEN - 1 + input_ids.shape[-1]

    # return (image_tensors, input_ids, info)
    return ([image], input_ids, info)


def first_fit_decreasing_with_ids(items, bin_capacity):
    # Create a list of (length, id) tuples and sort by length in descending order
    items = dict(sorted(items.items(), key=lambda x:x[1], reverse=True))
    
    bins = []  # List to store bins, each bin is a list of (length, id) tuples

    for item in items.items():
        # length, id = item
        id, length = item
        # Try to place the item in the first bin that can accommodate it
        placed = False
        for bin in bins:
            if sum(length for (id, length) in bin) + length <= bin_capacity:
                bin.append(item)
                placed = True
                break
        # If no bin could accommodate the item, create a new bin
        if not placed:
            bins.append([item])
    
    return bins

def preprocess_mint_pdf_interleaved(
    sample,
    tokenizer,
    image_processor,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """
    # image transform includes the padding process and normalization process

    tiff_image, info, key = sample
    info = json.loads(info)
    tiff_image = Image.open(BytesIO(tiff_image))

    info["key"] = key

    texts = info["texts"]

    assert tiff_image.n_frames == len(list(filter(None, info['images'])))
    
    # load images first to find which ones are valid
    image_tensors = []
    for i in range(tiff_image.n_frames):
        tiff_image.seek(i)
        page_image = tiff_image.copy()
        image = page_image.convert("RGB")
        if image.size[0] == 1 or image.size[1] == 1:
            continue
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensors.append(image_tensor)

    if len(image_tensors) == 0:
        return None

    # preprocess and pad images
    texts =  [DEFAULT_IMAGE_TOKEN+"\n" if x is None else x.replace(DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER)+END_CUNCK_TOKEN for x in texts]
    corpus = "".join(texts)

    input_ids, labels = tokenizer_image_token(corpus, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    if input_ids[0] != tokenizer.bos_token_id:
        input_ids = torch.cat([torch.LongTensor([tokenizer.bos_token_id]), input_ids])
        labels = torch.cat([torch.LongTensor([tokenizer.bos_token_id]), labels])

    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids = torch.cat([input_ids, torch.LongTensor([tokenizer.eos_token_id])])
        labels = torch.cat([labels, torch.LongTensor([tokenizer.eos_token_id])])
    
    # assert bos token
    assert input_ids[0] == tokenizer.bos_token_id
    assert len(image_tensors) == (input_ids == IMAGE_TOKEN_INDEX).sum()

    info['length'] = (NUM_IMAGES_TOKEN - 1) * len(image_tensors) + input_ids.shape[-1]

    return (image_tensors, input_ids, labels, info)


def concat_documents(documents, context_len, pad_token_id):

    new_input_ids = []
    # new_labels = []
    new_image_tensors = []
    new_length = []
    for (image_tensors, input_ids, length) in documents:
        new_image_tensors += image_tensors
        new_input_ids.append(input_ids)
        # new_labels.append(labels)
        new_length.append(length)

    new_input_ids = torch.cat(new_input_ids)

    if len(new_image_tensors) == 0:
        return None
    
    assert (new_input_ids == IMAGE_TOKEN_INDEX).sum() == len(new_image_tensors)
    assert (sum(new_length) - new_input_ids.shape[-1]) == (NUM_IMAGES_TOKEN - 1) * len(new_image_tensors)

    if new_input_ids.shape[-1] < context_len:
        new_input_ids = torch.cat([new_input_ids, torch.full((context_len - new_input_ids.shape[-1],), pad_token_id, dtype=new_input_ids.dtype)])
        # new_labels = torch.cat([new_labels, torch.full((context_len - new_labels.shape[-1],), IGNORE_INDEX, dtype=new_labels.dtype)])
    
    return {"image_tensors": new_image_tensors, "input_ids": new_input_ids, "lengths": new_length}

def main(args, gpu_id=0):
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    # if hasattr(tokenizer.config, "model_max_length"):
    #     context_len = tokenizer.config.model_max_length
    # else:
    context_len = 4096

    if "qwen" in args.model_path.lower():
        tokenizer.add_special_tokens({"additional_special_tokens": [END_CUNCK_TOKEN, "<s>", "<|pad|>"]})
        tokenizer.pad_token = "<|pad|>"
        tokenizer.bos_token = "<s>"
    elif "phi" in args.model_path.lower():
        tokenizer.add_special_tokens({"additional_special_tokens": [END_CUNCK_TOKEN]})
        tokenizer.pad_token = tokenizer.unk_token
    elif "llama" in args.model_path.lower():
        tokenizer.add_special_tokens({"additional_special_tokens": [END_CUNCK_TOKEN, "<|pad|>"]})
        tokenizer.pad_token = "<|pad|>"

    # set padding side to `left` for batch text generation
    tokenizer.padding_side = "right"
    
    if "mint" not in args.tar_file_path:
        for tar_id in list(range(12000))[gpu_id * args.tars_per_gpu: (gpu_id + 1) * args.tars_per_gpu]:
            if "ccs" in args.tar_file_path or "laion" in args.tar_file_path:
                shard_path = args.tar_file_path + "/{:05d}.tar".format(tar_id)
            elif "mmc4" in args.tar_file_path:
                shard_path = args.tar_file_path + "/shard_{}.tar".format(tar_id)
                if not os.path.exists(shard_path):# or not os.path.exists(shard_path.replace(".tar", "_stats.json")):
                    continue
            else:
                shard_path = args.tar_file_path + "/{:08d}.tar".format(tar_id)
            
            if os.path.exists(os.path.join(args.save_path, f'{tar_id}_stats.json')):
                logger.info(f"Tar {tar_id} processed")
                continue
            
            logger.info(f"Start processing tar {tar_id}")
            if not os.path.exists(os.path.join(args.save_path, str(tar_id))):
                os.mkdir(os.path.join(args.save_path, str(tar_id)))
            else:
                try:
                    shutil.rmtree(os.path.join(args.save_path, str(tar_id)))
                    print(f"Deleted folder and all its contents: {os.path.join(args.save_path, str(tar_id))}")
                except Exception as e:
                    print(f"Failed to delete folder. Reason: {e}")
                os.mkdir(os.path.join(args.save_path, str(tar_id)))
            
            if "obelics" in args.tar_file_path.lower() or "mmc4" in args.tar_file_path.lower():
                preprocess_fn = functools.partial(
                    preprocess_obelics_interleaved,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                )
                pipeline = [
                    wds.SimpleShardList(shard_path),
                    wds.split_by_worker,
                    wds.tarfile_to_samples(),
                    wds.to_tuple("__key__", "json"),
                    wds.map(preprocess_fn),
                    # wds.batched(args.batch_size, partial=True),
                ]
                dataset = wds.DataPipeline(*pipeline)
            else: # mainly uses this part
                preprocess_fn = functools.partial(
                    preprocess_caption,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                )
                pipeline = [
                    wds.SimpleShardList(shard_path),
                    wds.split_by_worker,
                    wds.tarfile_to_samples(),
                    wds.decode("pilrgb"),
                    wds.rename(image="jpg;png;jpeg;webp", text="txt", json="json"),
                    wds.to_tuple("image", "text", "json"),
                    wds.map(preprocess_fn),
                ]
                dataset = wds.DataPipeline(*pipeline)

            dataset = wds.WebLoader(
                dataset,
                collate_fn=None,
                batch_size=None,
                shuffle=False,
                num_workers=args.workers,
                persistent_workers=args.workers > 0,
            )
            
            final_data = {}
            length_to_uid = {}
            
            for data_tuple in tqdm(dataset):
                if not data_tuple:
                    continue
                image_tensors, input_ids, info = data_tuple 
                final_data[info['key']] = (image_tensors, input_ids, info['length'])
                length_to_uid[info['key']] = info['length']
            
            bins = first_fit_decreasing_with_ids(length_to_uid, context_len)

            total_len = []
            for bin in bins:
                bin_data = [final_data[key] for key, length in bin]
                concat_pkl = concat_documents(bin_data, context_len, tokenizer.pad_token_id)
                if not concat_pkl:
                    continue
                total_len.append(sum(concat_pkl['lengths']))
                key_str = uuid.uuid4().hex
                with open(os.path.join(os.path.join(args.save_path,  str(tar_id)), f'{key_str}.pkl'), 'wb') as f:
                    pickle.dump(concat_pkl, f)
            
            with open(os.path.join(args.save_path, f'{tar_id}_stats.json'), 'w') as f:
                f.write(json.dumps({"len": total_len}, indent=2))
                
            del final_data
    
    else:
        for shard_path in list(Path(args.tar_file_path).iterdir())[gpu_id * args.tars_per_gpu: (gpu_id + 1) * args.tars_per_gpu]:
            if shard_path.suffix != ".tar":
                continue
            # shard_path = args.tar_file_path + "/{:08d}.tar".format(tar_id)
            tar_id = "".join(str(shard_path.name).split("_")[2:4])

            if os.path.exists(os.path.join(args.save_path, f'{tar_id}_stats.json')):
                logger.info(f"Tar {tar_id} processed")
                continue
            
            logger.info(f"Start processing tar {tar_id}")
            if not os.path.exists(os.path.join(args.save_path, str(tar_id))):
                os.mkdir(os.path.join(args.save_path, str(tar_id)))
            else:
                try:
                    shutil.rmtree(os.path.join(args.save_path, str(tar_id)))
                    print(f"Deleted folder and all its contents: {os.path.join(args.save_path, str(tar_id))}")
                except Exception as e:
                    print(f"Failed to delete folder. Reason: {e}")
                os.mkdir(os.path.join(args.save_path, str(tar_id)))
            
            preprocess_fn = functools.partial(
                preprocess_mint_pdf_interleaved,
                tokenizer=tokenizer,
                image_processor=image_processor,
            )
            pipeline = [
                wds.SimpleShardList(str(shard_path)),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                # wds.decode("pilrgb"),
                wds.rename(image="tiff", json="json", key="__key__"),
                wds.to_tuple("image", "json", "key"),
                wds.map(preprocess_fn),
            ]
            dataset = wds.DataPipeline(*pipeline)

            dataset = wds.WebLoader(
                dataset,
                collate_fn=None,
                batch_size=None,
                shuffle=False,
                num_workers=args.workers,
                persistent_workers=args.workers > 0,
            )
            
            final_data = {}
            length_to_uid = {}
            
            for image_tensors, input_ids, labels, info in tqdm(dataset):
                
                final_data[info['key']] = (image_tensors, input_ids, labels, info['length'])
                length_to_uid[info['key']] = info['length']
            
            bins = first_fit_decreasing_with_ids(length_to_uid, context_len)

            total_len = []
            for bin in bins:
                bin_data = [final_data[key] for key, length in bin]
                concat_pkl = concat_documents(bin_data, context_len, tokenizer.pad_token_id)
                total_len.append(sum(concat_pkl['lengths']))
                key_str = uuid.uuid4().hex
                with open(os.path.join(os.path.join(args.save_path,  str(tar_id)), f'{key_str}.pkl'), 'wb') as f:
                    pickle.dump(concat_pkl, f)
            
            with open(os.path.join(args.save_path, f'{tar_id}_stats.json'), 'w') as f:
                f.write(json.dumps({"len": total_len}, indent=2))
                
            del final_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--tar-file-path", type=str, default="Open-Qwen2VL-Data")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tars-per-gpu", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-len", type=int, default=4096)
    args = parser.parse_args()
    logger.info(args)
    main(args, gpu_id=args.gpu_id)