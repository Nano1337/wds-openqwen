"""
wds_dataset.py

Implements a dataset class that can load sequence-packed data from WebDataset format.
"""
import os
import io
import torch
import pickle
import logging
import webdataset as wds
from typing import Dict, List, Type, Tuple, Optional
from pathlib import Path

from transformers import PreTrainedTokenizerBase
from prismatic.preprocessing.transforms import ImageTransform
from prismatic.preprocessing.prompts import PromptBuilder
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX

logger = logging.getLogger(__name__)

class WDSPackedDataset:
    """Dataset for loading pre-packed sequences from WebDataset tar files for VLM training."""
    
    def __init__(
        self,
        shards_path: str,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        train_num_samples: int,
        workers: int = 4,
        shuffle_buffer: int = 1000,
    ) -> None:
        """
        Initialize a WebDataset for packed sequences.
        
        Args:
            shards_path: Path to the shards.txt file or directory containing tar files
            image_transform: Image transformation function
            tokenizer: Tokenizer for text processing
            prompt_builder_fn: Function to build prompts
            train_num_samples: Number of samples for training
            workers: Number of workers for data loading
            shuffle_buffer: Size of shuffle buffer
        """
        super().__init__()
        self.shards_path = shards_path
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.train_num_samples = train_num_samples
        self.workers = workers
        self.shuffle_buffer = shuffle_buffer
        self.dataset_type = "pretrain"
        
        # Load list of shards
        if os.path.isfile(shards_path) and shards_path.endswith(".txt"):
            with open(shards_path, "r") as f:
                self.shard_list = [line.strip() for line in f.readlines()]
        else:
            # If it's a directory, find all tar files
            if os.path.isdir(shards_path):
                self.shard_list = [
                    os.path.join(shards_path, f) 
                    for f in os.listdir(shards_path) 
                    if f.startswith("packed_") and f.endswith(".tar")
                ]
            else:
                # If it's a single shard or a pattern
                self.shard_list = [shards_path]
        
        logger.info(f"Found {len(self.shard_list)} shards in {shards_path}")
    
    def create_dataset(self):
        """Create a WebDataset pipeline for loading packed sequences."""
        
        # Define a function to process each sample
        def process_sample(sample):
            # Load pickle data from bytes
            data_pkl = pickle.loads(sample["pkl"])
            
            # Apply image transformations
            image_tensors = [self.image_transform(image) for image in data_pkl['image_tensors']]
            data_pkl['image_tensors'] = image_tensors
            
            # Create pixel_values from image_tensors - stack them for batching
            if len(image_tensors) > 0:
                data_pkl['pixel_values'] = torch.stack(image_tensors)
            
            # Create labels by masking out image tokens
            labels = data_pkl['input_ids'].clone()
            labels[labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
            data_pkl['labels'] = labels
            
            return data_pkl
        
        # Create WebDataset pipeline
        dataset = (
            wds.WebDataset(self.shard_list)
            .shuffle(self.shuffle_buffer)
            .decode()
            .map(process_sample)
            .to_tuple("pixel_values", "input_ids", "labels", "lengths")
        )
        
        return dataset

    def get_dataloader(self, batch_size=1):
        """Create a dataloader from the WebDataset."""
        dataset = self.create_dataset()
        
        # Create WebLoader
        dataloader = wds.WebLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Already shuffled in the pipeline
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
        )
        
        # Set approximate length - needed for epoch tracking
        dataloader.num_batches = self.train_num_samples // batch_size
        dataloader.num_samples = self.train_num_samples
        
        return dataloader
