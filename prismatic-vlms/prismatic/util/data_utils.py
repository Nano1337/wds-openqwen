"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

"""
Util functions for initializing webdataset objects
"""

import ast
import json
import logging
import os
import random
import sys
from multiprocessing import Value
from pathlib import Path

import braceexpand
import numpy as np
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            # Weizhi: made a change here, I did not understand why LLaVA and Prismatic-VLM all adopted this dummy images
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
            # pixel_values = None
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )
    
@dataclass
class PaddedCollatorForMMLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        pixel_values = [instance["image_tensors"] for instance in instances]
        # attention_mask = torch.stack([instance["attention_mask"] for instance in instances])
        
        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            # attention_mask=None,
            labels=labels,
        )


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):

    # if isinstance(shards, Path):
    #     files_or_subdirs = list(shards.iterdir())
    #     shards_list = []
    #     for file_or_subdir in files_or_subdirs:
    #         if file_or_subdir.suffix == ".tar":
    #             shards_list.append(str(file_or_subdir))
    #             # shard = file_or_subdir.with_suffix("")
    #             # prefix, suffix = split_filename(pattern, shard)
    # else:
    #     
    shards_list = list(braceexpand.braceexpand(shards))

    dir_path = os.path.dirname(str(shards))
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum(
            [
                int(sizes[os.path.basename(shard)])
                if os.path.basename(shard) in sizes
                else 0
                for shard in shards_list
            ]
        )
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        # Skip invalid file samples without expected keys
        if not isinstance(filesample, dict):
            logging.warning(f"Skipping non-dict filesample: {filesample}")
            continue
        if "fname" not in filesample or "data" not in filesample:
            logging.warning(f"Skipping malformed filesample missing 'fname' or 'data': {filesample}")
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    labels = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
        labels.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    
    for x in insert_separator(prompt_chunks, [IGNORE_INDEX] * (offset + 1)):
        labels.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids, labels