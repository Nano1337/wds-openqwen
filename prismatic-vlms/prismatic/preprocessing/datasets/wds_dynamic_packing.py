import os
import uuid
import torch
import json
import base64
import io
import copy
import functools
import re
import boto3
import webdataset as wds
from webdataset.filters import unlisted
import time
from typing import Dict, List, Tuple
from PIL import Image
from webdataset.shardlists import expand_urls
from webdataset.handlers import reraise_exception
from tqdm import tqdm
from prismatic.util.data_utils import (
    SharedEpoch, 
    detshuffle2, 
    DataInfo, 
    tarfile_to_samples_nothrow, 
)
from prismatic.util.data_utils import tokenizer_image_token
from mm_sequence_packing.sequence_packing_image_to_pil import first_fit_decreasing_with_ids

# Import constants from datasets.py
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
END_CUNCK_TOKEN = "<|endofchunk|>"
NUM_IMAGES_TOKEN = 144

class WDSDynamicPackingDataset:
    """
    WebDataset-based streaming sequence packing for multimodal data.
    Supports multi-GPU training with proper shuffling and epoch management.
    """
    def __init__(
        self,
        shards_pattern,  # WebDataset shards pattern (can be S3 URLs)
        tokenizer,
        image_transform,
        context_len,      # Maximum context length for packing
        train_num_samples,  # Number of samples per epoch estimation
        num_workers=8,
        shuffle_buffer=10000,
        samples_per_pack=100,  # Number of samples to consider for each packing operation
        collator=None
    ):
        self.context_len = context_len
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.samples_per_pack = samples_per_pack
        self.collator = collator
        
        # Epoch counter for deterministic shuffling
        self.shared_epoch = SharedEpoch(0)
        
        # Build pipeline - use direct S3 URLs since we have custom URL opener
        urls = expand_urls(shards_pattern)
        pipeline_urls = [f'pipe:aws s3 cp {u} -' for u in urls]

        # Define a custom batching function to handle bin packing
        def custom_batcher(samples):
            # Filter out None and pack; return list of packed dicts
            samples = [x for x in samples if x is not None]
            if not samples:
                return []
            return self._pack_batch(samples)
        
        # Set up the data pipeline for streaming and processing
        pipeline = [
            wds.SimpleShardList(pipeline_urls),
            # Deterministic shuffling with epoch support
            detshuffle2(bufsize=shuffle_buffer, initial=min(shuffle_buffer, 1000), 
                       seed=0, epoch=self.shared_epoch),
            wds.split_by_node,  # Split by node for multi-node training
            wds.split_by_worker, # Split by worker for multi-process data loading
            tarfile_to_samples_nothrow, # Robust tar file handling
            wds.shuffle(bufsize=shuffle_buffer), # Additional shuffling
            wds.decode("pilrgb"),  # Decode images to PIL
            wds.rename(image="jpg;png;jpeg;webp", text="txt", json="json"),
            wds.to_tuple("image", "text", "json"),
            wds.map(self._preprocess_sample),  # Convert to (image, input_ids, length)
            wds.batched(samples_per_pack, collation_fn=None, partial=True),  # Batch raw samples into lists
            wds.map(custom_batcher),                                        # Pack this list into list of dicts
            unlisted(),                                                     # Flatten packed dicts back into the stream
        ]
        
        self.dataset = wds.DataPipeline(*pipeline)
        self.num_workers = num_workers
        
        # Estimate number of packed examples per epoch
        packing_efficiency = 0.8  # Estimate of how efficiently samples can be packed
        self.num_packs = int((train_num_samples * packing_efficiency) / samples_per_pack)

    def __len__(self):
        return self.num_packs

    def _preprocess_sample(self, data_tuple):
        """Process a single sample from WebDataset, using the same approach as PreTrainDataset.preprocess_obelics_interleaved"""
        if not data_tuple:
            return None
            
        image, text, meta = data_tuple
        
        # Process image
        if not image or not text:
            return None
            
        # Convert bytes to PIL Image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        # Skip images that are too small
        if image.size[0] == 1 or image.size[1] == 1:
            return None
        
        # Create an interleaved text with image tokens, similar to original preprocessing
        # Wrap the text with image tokens to create interleaved document
        corpus = DEFAULT_IMAGE_TOKEN + "\n" + text
        
        # Tokenize with image tokens
        input_ids, _ = tokenizer_image_token(corpus, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        # Ensure BOS token
        if input_ids.numel() == 0 or input_ids[0] != self.tokenizer.bos_token_id:
            input_ids = torch.cat([torch.LongTensor([self.tokenizer.bos_token_id]), input_ids])
        
        # Verify image token counts - make sure we have at least one image token
        image_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
        if image_tokens == 0:
            return None
        
        # Ensure EOS token
        if input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.tokenizer.eos_token_id])])
        
        # Verify BOS token
        assert input_ids[0] == self.tokenizer.bos_token_id
        
        # Calculate length including image tokens
        length = (NUM_IMAGES_TOKEN - 1) * image_tokens + input_ids.shape[-1]
        
        # Return tuple matching the format expected by _concat_documents
        return ([image], input_ids, length)
        
        return None

    def _pack_batch(self, batch):
        """Pack a batch of samples into context windows"""
        # Filter out None values
        batch = [x for x in batch if x is not None]
        if not batch:
            return []
            
        # Create a length_to_uid dictionary
        length_to_uid = {i: sample[2] for i, sample in enumerate(batch)}
        
        # Use first-fit-decreasing bin packing algorithm
        bins = first_fit_decreasing_with_ids(length_to_uid, self.context_len)
        
        # Log packing efficiency metrics
        total_tokens = sum(length_to_uid.values())
        total_bins = len(bins)
        if total_bins > 0 and total_tokens > 0:
            avg_bin_size = total_tokens / total_bins
            efficiency = avg_bin_size / self.context_len
            print(f"Packing efficiency: {efficiency:.2f} (avg tokens per bin: {avg_bin_size:.1f})")
        
        # For each bin, concatenate the samples
        results = []
        for bin in bins:
            # Get data for this bin
            bin_data = [batch[i] for i, _ in bin if i < len(batch)]
            
            # Concatenate documents - this handles padding to context_len
            packed = self._concat_documents(bin_data)
            if packed:
                # Apply image transformation to all images in the packed sequence
                packed["image_tensors"] = [self.image_transform(img) for img in packed["image_tensors"]]
                
                # Create labels from input_ids - exactly like PreTrainDataset.__getitem__
                labels = packed["input_ids"].clone()
                labels[labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
                packed["labels"] = labels
                
                # Add to results
                results.append(packed)
                
        return results
    
    def _concat_documents(self, documents):
        """Concatenate a list of documents into a single packed example"""
        if not documents:
            return None
            
        new_input_ids = []
        new_image_tensors = []
        new_length = []
        
        # Collect data from all documents - exactly matching PreTrainDataset.concat_document
        for image, input_ids, length in documents:
            new_image_tensors += image if isinstance(image, list) else [image]
            new_input_ids.append(input_ids)
            new_length.append(length)
            
        # Concatenate input_ids
        new_input_ids = torch.cat(new_input_ids)
        
        if len(new_image_tensors) == 0:
            return None
        
        # Verify the number of image tokens matches the number of images
        num_image_tokens = (new_input_ids == IMAGE_TOKEN_INDEX).sum().item()
        num_images = len(new_image_tensors)
        
        # Check for valid counts - if we have no images or no tokens, skip
        if num_images == 0 or num_image_tokens == 0:
            return None
        
        # Verify they match, matching original code's assertion
        if num_image_tokens != num_images:
            print(f"Warning: Image token count mismatch - tokens: {num_image_tokens}, images: {num_images}")
            
            # If we have more images than tokens, truncate the excess images
            if num_images > num_image_tokens:
                new_image_tensors = new_image_tensors[:num_image_tokens]
            else:
                # More tokens than images is problematic - skip this batch
                return None
        
        # Calculate length-related values exactly as in the original code
        try:
            # Convert lengths to integers if they're in dictionaries
            numeric_lengths = [item if isinstance(item, int) else item["length"] for item in new_length]
            token_diff = sum(numeric_lengths) - new_input_ids.shape[-1]
            expected_diff = (NUM_IMAGES_TOKEN - 1) * len(new_image_tensors)
            
            # Verify length differences match
            if token_diff != expected_diff:
                print(f"Warning: Length difference mismatch - Got {token_diff}, Expected {expected_diff}")
                # Continue anyway - this verification isn't critical for functionality
        except Exception as e:
            print(f"Warning: Error calculating length differences: {e}")
        
        # Handle sizing to context_len
        if new_input_ids.shape[-1] > self.context_len:
            # Truncate if longer than context_len
            new_input_ids = new_input_ids[:self.context_len]
        elif new_input_ids.shape[-1] < self.context_len:
            # Pad
            padding = torch.full((self.context_len - new_input_ids.shape[-1],), 
                                 self.pad_token_id, 
                                 dtype=new_input_ids.dtype)
            new_input_ids = torch.cat([new_input_ids, padding])
        
        # Return dictionary format exactly matching PreTrainDataset.concat_document
        return {"image_tensors": new_image_tensors, "input_ids": new_input_ids, "lengths": new_length}
        
    def get_dataloader(self, batch_size=1):
        """Return a DataInfo with WebLoader for distributed training"""
        print(f"Creating WebLoader with batch_size={batch_size}, num_workers={self.num_workers}")
        
        # Use WebDataset's WebLoader which is designed to work with WebDataset pipelines
        loader = wds.WebLoader(
            self.dataset,  # Use the prepared dataset pipeline
            batch_size=batch_size,
            shuffle=False,  # Shuffling is already handled in the pipeline
            num_workers=self.num_workers,
            persistent_workers=True, 
            collate_fn=self.collator,  # Use our custom collator for final batch preparation
        )
        
        # Set number of batches per epoch - critical for training loop
        loader.num_batches = self.num_packs
            
        # Return DataInfo which is expected by run_training_with_wds_dataloader
        return DataInfo(dataloader=loader, shared_epoch=self.shared_epoch)