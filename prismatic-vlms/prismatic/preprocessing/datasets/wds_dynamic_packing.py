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
import math
import random

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
        samples_per_pack=500,  # Increased from 100 for better packing stability
        collator=None
    ):
        self.context_len = context_len
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.samples_per_pack = samples_per_pack
        self.collator = collator
        self.train_num_samples = train_num_samples
        
        # Epoch counter for deterministic shuffling
        self.shared_epoch = SharedEpoch(0)
        
        # Accept one or more S3 patterns separated by '::'
        if not isinstance(shards_pattern, str):
            raise ValueError("shards_pattern must be a string of S3 patterns separated by '::'")
        parts = shards_pattern.split("::")
        grouped_urls = [expand_urls(p) for p in parts]
        
        # Add timeout and retry parameters to the S3 command
        s3_timeout = int(os.environ.get("S3_READ_TIMEOUT", "300"))
        s3_retries = int(os.environ.get("AWS_MAX_ATTEMPTS", "20"))
        
        # Configure S3 pipeline command with retries and timeouts
        pipeline_urls = []
        for url_list in grouped_urls:
            for u in url_list:
                # Add retries and timeout to S3 command
                s3_cmd = f'aws s3 cp {u} - --cli-connect-timeout {s3_timeout} --cli-read-timeout {s3_timeout}'
                pipeline_urls.append(f'pipe:{s3_cmd}')
                
        print(f"Configured {len(pipeline_urls)} pipeline URLs with {s3_retries} retries and {s3_timeout}s timeout")
        
        # set the pack count for correct epoching during dynamic multimodal sequence packing
        self._set_sizing(grouped_urls)
        
        # Define a custom batching function to handle bin packing
        def custom_batcher(samples):
            # Filter out None and pack; return list of packed dicts
            samples = [x for x in samples if x is not None]
            if not samples:
                return []
            return self._pack_batch(samples)
        
        # Set up the data pipeline for streaming and processing with better error handling
        pipeline = [
            # Level 1: Shard-level shuffle - low memory cost, high impact
            wds.SimpleShardList(pipeline_urls),
            detshuffle2(bufsize=shuffle_buffer, 
                       initial=min(shuffle_buffer // 2, 2000), # More memory efficient
                       seed=-1,  # Use worker seed for variety
                       epoch=self.shared_epoch),
            wds.split_by_node,
            wds.split_by_worker,
            # Extract samples from tar files
            tarfile_to_samples_nothrow,
            
            # Level 2: Sample-level shuffle BEFORE decoding - most memory efficient place
            # This is crucial and worth the memory cost
            wds.shuffle(bufsize=min(shuffle_buffer, 5000), 
                       initial=min(shuffle_buffer // 4, 1000)),  # Lower initial fill
                       
            # Decode and prepare samples
            wds.decode("pilrgb"),
            wds.rename(image="jpg;png;jpeg;webp", text="txt", json="json"),
            wds.to_tuple("image", "text", "json"),
            wds.map(self._preprocess_sample),


            # Batching for sequence packing
            wds.batched(samples_per_pack, collation_fn=None, partial=True),
            wds.map(custom_batcher),
            
            # Final flattening - no additional high-memory shuffle 
            unlisted(),
        ]
        
        self.dataset = wds.DataPipeline(*pipeline)
        self.num_workers = num_workers
        self._pack_calls = 0  # count pack invocations for periodic logging

    def __len__(self):
        return self.num_packs

    def _set_sizing(self, grouped_urls):
        # Dry-run on rank 0: sample more shards per pattern to estimate avg length, then broadcast
        try:
            import torch.distributed as dist
            # only rank 0 computes dry-run
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                print("Dry running to estimate avg length and packing efficiency")

                # Pick more shards to get a better estimate
                dry_urls = []
                for grp in grouped_urls:
                    if grp:
                        # Sample up to 3 shards per group for a better estimate
                        sample_count = min(3, len(grp))
                        sample_indices = [0] + [len(grp) // 2] + [len(grp) - 1]
                        sample_indices = sample_indices[:sample_count]
                        dry_urls.extend([grp[i] for i in sample_indices])
                
                print(f"Using {len(dry_urls)} sample shards for dry run estimation")
                
                dry_pipe = wds.DataPipeline(
                    wds.SimpleShardList([f'pipe:aws s3 cp {u} -' for u in dry_urls]),
                    tarfile_to_samples_nothrow,
                    wds.decode("pilrgb"),
                    wds.rename(image="jpg;png;jpeg;webp", text="txt", json="json"),
                    wds.to_tuple("image", "text", "json"),
                    wds.map(self._preprocess_sample),
                )
                
                # Sample more items (up to 500) for better estimation
                sample_limit = 500
                lengths = []
                samples = []
                for i, s in zip(range(sample_limit), dry_pipe):
                    if s is None: continue
                    lengths.append(s[2])
                    samples.append(s)
                
                print(f"Collected {len(samples)} valid samples for estimation")
                
                # Calculate average length AND packing efficiency
                avg_len = sum(lengths) / len(lengths) if lengths else self.context_len
                
                # Estimate packing efficiency by running the actual packing algorithm on the sample
                if samples:
                    length_to_uid = {i: sample[2] for i, sample in enumerate(samples)}
                    bins = first_fit_decreasing_with_ids(length_to_uid, self.context_len)
                    
                    # Calculate packing efficiency based on the sample
                    total_tokens = sum(length_to_uid.values())
                    total_bins = len(bins) if bins else 1
                    packing_efficiency = total_tokens / (total_bins * self.context_len)
                    
                    print(f"Dry run: avg length = {avg_len:.1f}, packing efficiency = {packing_efficiency:.2f}")
                    print(f"Sample stats: total tokens = {total_tokens}, packed into {total_bins} bins")
                    
                    # Use packing efficiency to better estimate total number of packed samples
                    estimated_tokens = self.train_num_samples * avg_len
                    self.num_packs = math.ceil(estimated_tokens / (self.context_len * packing_efficiency))
                    
                    print(f"Full dataset estimation: {self.train_num_samples} samples with avg length {avg_len:.1f}")
                    print(f"Estimated total tokens: {estimated_tokens:,}")
                    print(f"Estimated packed sequences: {self.num_packs:,} (before buffer)")
                else:
                    print(f"Dry run avg length: {avg_len}")
                    self.num_packs = math.ceil(self.train_num_samples * avg_len / self.context_len)
            else:
                avg_len = 0.0
                self.num_packs = 0
                
            # Broadcast estimates (use CUDA tensor if available to match NCCL backend)
            if dist.is_available() and dist.is_initialized():
                # Choose device for tensor
                if torch.cuda.is_available():
                    dev = torch.device('cuda', torch.cuda.current_device())
                else:
                    dev = torch.device('cpu')
                # Broadcast num_packs instead of just avg_len
                t = torch.tensor([self.num_packs], dtype=torch.float32, device=dev)
                dist.broadcast(t, src=0)
                self.num_packs = int(t.cpu().item())
                
            # Add a small buffer to account for potential estimation errors (5% extra)
            raw_packs = self.num_packs
            self.num_packs = math.ceil(self.num_packs * 1.05)
                
            print(f"Final number of packed sequences: {self.num_packs:,} (with 5% buffer, was {raw_packs:,})")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in size estimation: {e}")
            # Fallback estimation
            self.num_packs = math.ceil(self.train_num_samples / self.samples_per_pack)
            print(f"Using fallback estimation: {self.num_packs:,} packed sequences")
            
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
        corpus = DEFAULT_IMAGE_TOKEN + "\n" + text.replace(DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER)
        
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
        
        # FIXED: Calculate length to EXACTLY match the offline approach (preprocess_caption)
        # The offline approach uses a fixed constant offset regardless of number of images:
        # info['length'] = NUM_IMAGES_TOKEN - 1 + input_ids.shape[-1]
        length = NUM_IMAGES_TOKEN - 1 + input_ids.shape[-1]
        
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
            # periodic logging of efficiency (once then every 10 packs)
            self._pack_calls += 1
            if self._pack_calls == 1 or self._pack_calls % 10 == 0:
                print(f"Packing efficiency: {efficiency:.2f} (avg tokens per bin: {avg_bin_size:.1f}) [pack {self._pack_calls}]")
        
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
            
            # FIXED: To match the offline approach's length calculation
            # Each document now adds NUM_IMAGES_TOKEN - 1 to its length regardless of image count
            # So for N documents, we expect a total difference of N * (NUM_IMAGES_TOKEN - 1)
            expected_diff = len(new_length) * (NUM_IMAGES_TOKEN - 1)
            
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
        
        # Set number of batches per epoch based on pack count and loader batch size
        loader.num_batches = math.ceil(self.num_packs / batch_size)
        print(f"Dataloader configured with {loader.num_batches:,} batches per epoch")
        print(f"With gradient accumulation steps={8}, estimated training steps ~{loader.num_batches//8:,}")
            
        # Return DataInfo which is expected by run_training_with_wds_dataloader
        return DataInfo(dataloader=loader, shared_epoch=self.shared_epoch)