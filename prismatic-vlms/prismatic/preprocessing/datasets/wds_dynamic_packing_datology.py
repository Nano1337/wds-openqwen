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
import torch.distributed as dist
from webdataset.handlers import reraise_exception
from tqdm import tqdm
from prismatic.util.data_utils import (
    SharedEpoch,
    detshuffle2,
    DataInfo,
    tarfile_to_samples_nothrow,
)
from prismatic.util.data_utils import tokenizer_image_token, log_and_continue
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

# Caption separator constant for template captions
CAPTION_SEPARATOR = "|\uFFFC|"

class LinearScheduler:
    """Simple linear scheduler for controlling synthetic caption ratio"""
    def __init__(self, start_value, end_value, total_steps, current_step=0, is_dynamic=True, clip_score_threshold=0.28):
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = max(1, total_steps)
        self.current_step = current_step
        self.is_dynamic = is_dynamic
        self.clip_score_threshold = clip_score_threshold

    def step(self):
        if self.is_dynamic:
            self.current_step += 1

    def get_value(self):
        if self.is_dynamic:
            # Linear interpolation between start_value and end_value
            progress = min(1.0, self.current_step / self.total_steps)
            return self.start_value + (self.end_value - self.start_value) * progress
        else:
            return self.end_value

    def get_clipscore_threshold(self):
        return self.clip_score_threshold

class WDSDynamicPackingDatologyDataset:
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
        collator=None,
        start_synthetic_ratio=0.5, # Default value for synthetic caption sampling
        end_synthetic_ratio=0.8,   # Default end value for synthetic caption sampling
        dynamic_data_scheduler=True, # Whether to dynamically change the synthetic ratio
        clip_score_threshold=0.28   # Threshold for filtering synthetic captions
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
            
        # Centralize URL expansion - only rank 0 expands, others receive via broadcast
        pipeline_urls = self._centralized_url_expansion(shards_pattern)

        print(f"Pipeline configured with {len(pipeline_urls)} URLs")

        self.num_packs = 1  # Temporary value for scheduler init
        if start_synthetic_ratio is not None and end_synthetic_ratio is not None:
            self.data_scheduler = LinearScheduler(
                start_synthetic_ratio,
                end_synthetic_ratio,
                self.num_packs,  # Will be updated after sizing
                0,
                is_dynamic=dynamic_data_scheduler,
                clip_score_threshold=clip_score_threshold
            )
        else:
            self.data_scheduler = None

        # set the pack count for correct epoching during dynamic multimodal sequence packing
        self._set_sizing(pipeline_urls)

        # Update data_scheduler with correct num_packs after sizing
        if self.data_scheduler is not None:
            self.data_scheduler.total_steps = self.num_packs

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
            detshuffle2(
                bufsize=shuffle_buffer,
                initial=min(shuffle_buffer // 2, 2000),
                seed=-1,
                epoch=self.shared_epoch
            ),
            wds.split_by_node,
            wds.split_by_worker,

            # Extract samples from tar files
            tarfile_to_samples_nothrow,

            # Level 2: Sample-level shuffle BEFORE decoding - most memory efficient place
            # This is crucial and worth the memory cost
            wds.shuffle(
                bufsize=min(shuffle_buffer, 5000),
                initial=min(shuffle_buffer // 4, 1000)
            ),

            # Decode and prepare samples
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt.json", json="json", handler=log_and_continue),
            wds.to_tuple("image", "text", "json", handler=log_and_continue),
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

    def _centralized_url_expansion(self, shards_pattern):
        """Centralize URL expansion to rank 0 and broadcast results to other ranks"""
        # Add timeout and retry parameters to the S3 command
        s3_timeout = int(os.environ.get("S3_READ_TIMEOUT", "300"))
        s3_retries = int(os.environ.get("AWS_MAX_ATTEMPTS", "20"))
        
        pipeline_urls = []
        is_distributed = False
        
        try:
            # Check if we're in distributed mode
            is_distributed = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if is_distributed else 0
            
            # Only rank 0 expands URLs
            if not is_distributed or rank == 0:
                start_time = time.time()
                parts = shards_pattern.split("::")
                print(f"Rank {rank}: expanding urls: {parts}")
                
                # expand urls
                grouped_urls = [expand_urls(p) for p in parts]
                
                # Configure S3 pipeline command with retries and timeouts
                print(f"Rank {rank}: configuring pipeline urls")
                for url_list in grouped_urls:
                    for u in url_list:
                        s3_cmd = f'aws s3 cp {u} - --cli-connect-timeout {s3_timeout} --cli-read-timeout {s3_timeout}'
                        pipeline_urls.append(f'pipe:{s3_cmd}')
                
                print(f"Rank {rank}: URL expansion + configuration completed in {time.time() - start_time:.2f}s")
                print(f"Configured {len(pipeline_urls)} pipeline URLs with {s3_retries} retries and {s3_timeout}s timeout")
            
            # Broadcast the URLs directly using broadcast_object_list
            if is_distributed:
                # Create a list with a single element for broadcasting
                obj_list = [pipeline_urls] if rank == 0 else [None]
                dist.broadcast_object_list(obj_list, src=0)
                pipeline_urls = obj_list[0]
                
                if rank != 0:
                    print(f"Rank {rank}: Received {len(pipeline_urls)} pipeline URLs via broadcast")
                
        except Exception as e:
            # Fallback in case of errors
            import traceback
            traceback.print_exc()
            print(f"Error in URL expansion centralization: {e}")
            
            # Individual node fallback
            if len(pipeline_urls) == 0:
                parts = shards_pattern.split("::")
                print(f"Fallback: expanding urls individually: {parts}")
                grouped_urls = [expand_urls(p) for p in parts]
                
                for url_list in grouped_urls:
                    for u in url_list:
                        s3_cmd = f'aws s3 cp {u} - --cli-connect-timeout {s3_timeout} --cli-read-timeout {s3_timeout}'
                        pipeline_urls.append(f'pipe:{s3_cmd}')
                        
                print(f"Fallback: Configured {len(pipeline_urls)} pipeline URLs")
            
        return pipeline_urls

    def _set_sizing(self, pipeline_urls):
        # Dry-run on rank 0: sample more shards per pattern to estimate avg length, then broadcast
        try:
            import torch.distributed as dist
            # only rank 0 computes dry-run
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                print("Dry running to estimate avg length and packing efficiency")

                # Pick shards from pipeline_urls for dry run
                sample_count = min(3, len(pipeline_urls))
                if sample_count > 0:
                    # Pick from beginning, middle, and end
                    indices = [0]
                    if sample_count > 1:
                        indices.append(len(pipeline_urls) // 2)
                    if sample_count > 2:
                        indices.append(len(pipeline_urls) - 1)
                    
                    dry_urls = [pipeline_urls[i] for i in indices]
                    
                    print(f"Using {len(dry_urls)} sample shards for dry run estimation")
                    
                    dry_pipe = wds.DataPipeline(
                        wds.SimpleShardList(dry_urls),
                        tarfile_to_samples_nothrow,
                        wds.decode("pilrgb", handler=log_and_continue),
                        wds.rename(image="jpg;png;jpeg;webp", text="txt.json", json="json", handler=log_and_continue),
                        wds.to_tuple("image", "text", "json", handler=log_and_continue),
                        wds.map(self._preprocess_sample),
                    )
                    
                    # Sample items for better estimation
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
                    # Fallback if no pipeline URLs
                    print("No pipeline URLs for dry run, using fallback estimation")
                    self.num_packs = math.ceil(self.train_num_samples / self.samples_per_pack)
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
            print("Terminating run due to error in size estimation.")
            import sys
            sys.exit(1)
            
    def _preprocess_sample(self, data_tuple):
        """Process a single sample from WebDataset, using the same approach as PreTrainDataset.preprocess_obelics_interleaved"""
        if not data_tuple:
            return None

        image, text_json, meta = data_tuple

        if not image or not text_json:
            return None

        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        # Skip images that are too small
        if image.size[0] == 1 or image.size[1] == 1:
            return None

        try:
            captions, clip_scores = self.preprocess_text_sampling(text_json)
            text = self.select_caption(captions, clip_scores)

            if not text:
                return None
        except Exception as e:
            print(f"Error processing caption: {e}")
            return None

        # Create an interleaved text with image tokens, similar to original preprocessing
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

        # Calculate length to match the offline approach
        length = NUM_IMAGES_TOKEN - 1 + input_ids.shape[-1]

        # Return tuple matching the format expected by _concat_documents
        return ([image], input_ids, length)

    def preprocess_text_sampling(self, text, caption_separator=CAPTION_SEPARATOR):
        """
        Process text input from JSON files to extract captions and scores
        """
        if isinstance(text, bytes):
            text = text.decode('utf-8')

        if isinstance(text, dict):
            txt_loaded = text
        else:
            try:
                txt_loaded = json.loads(text)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[ERROR] JSON decode error: {e}")
                print(f"[ERROR] Offending sample (truncated to 500 chars): {repr(text)[:500]}")
                # Assume already decoded, possibly a long string
                if len(text) > 3000:
                    text = text[:3000]  # Will be cut by tokenizer context length anyway
                txt_loaded = text

        if isinstance(txt_loaded, dict):
            # Dictionary format, unpack it with special handling for our format
            if 'orig_cap' in txt_loaded and 'template_cap' in txt_loaded: 
                # Split template_cap if it contains multiple captions
                captions = [txt_loaded['orig_cap']]
                
                # Add template captions if available
                if txt_loaded['template_cap']:
                    captions.extend(txt_loaded['template_cap'].split(caption_separator))
                
                # Get scores if available
                caption_scores = None
                if 'orig_score' in txt_loaded and 'score' in txt_loaded:
                    caption_scores = [float(txt_loaded['orig_score'])]
                    if txt_loaded['score']:
                        caption_scores.extend([float(s) for s in txt_loaded['score'].split(' ')])
                
                return captions, caption_scores
            else:
                # Inconsistent keys, just return as a string
                return str(txt_loaded), None
        else:
            # Just return the text/list
            if isinstance(txt_loaded, str):
                return txt_loaded, None
            elif isinstance(txt_loaded, list):
                # Convert each entry to a str, then return
                return [str(x) for x in txt_loaded], None
            else:
                # Cast to a string for unsupported types
                return str(txt_loaded), None

    def select_caption(self, captions, clip_scores=None, filter_incomplete=True):
        """
        Select a caption from available options based on data scheduler
        """
        if isinstance(captions, str):
            return captions

        if not captions:
            return ""

        if len(captions) == 1:
            return captions[0]

        if self.data_scheduler is None:
            # No scheduler, just take the original caption
            return captions[0]

        # Get current synthetic caption probability
        synth_caption_prob = self.data_scheduler.get_value()
        clip_score_threshold = self.data_scheduler.get_clipscore_threshold()

        original_text = [captions[0]]
        synthetic_text = captions[1:]

        # Filter by clip scores if available
        if clip_scores is not None and len(clip_scores) > 1:
            valid_indices = []
            for i, score in enumerate(clip_scores[1:], start=0):
                if score > clip_score_threshold and i < len(synthetic_text):
                    valid_indices.append(i)
                else:
                    break
            synthetic_text = [synthetic_text[i] for i in valid_indices]

        # Filter incomplete sentences if requested
        if filter_incomplete and synthetic_text:
            filtered_synthetic = []
            for text in synthetic_text:
                # Keep if ending with period or long enough
                if text.strip().endswith('.') or len(text) >= 77:
                    filtered_synthetic.append(text)
            synthetic_text = filtered_synthetic

        # If we have no synthetic captions after filtering, use the original
        if not synthetic_text:
            return original_text[0]

        # Calculate sampling weights with exponential decay
        n_synth = len(synthetic_text)
        synth_probs = [1/(2**x) for x in range(1, n_synth+1)]
        synth_probs = [(x * synth_caption_prob) / sum(synth_probs) for x in synth_probs]
        probs = [1 - synth_caption_prob] + synth_probs

        # Sample based on weights
        selected = random.choices(original_text + synthetic_text, weights=probs, k=1)[0]

        # Step the data scheduler
        if self.data_scheduler is not None:
            self.data_scheduler.step()

        return selected
    
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
            self._pack_calls += 1
            if self._pack_calls == 1 or self._pack_calls % 10 == 0:
                print(f"Packing efficiency: {efficiency:.2f} (avg tokens per bin: {avg_bin_size:.1f}) [pack {self._pack_calls}]")

        results = []
        for bin in bins:
            bin_data = [batch[i] for i, _ in bin if i < len(batch)]
            packed = self._concat_documents(bin_data)
            if packed:
                packed["image_tensors"] = [self.image_transform(img) for img in packed["image_tensors"]]
                labels = packed["input_ids"].clone()
                labels[labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
                packed["labels"] = labels
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

        new_input_ids = torch.cat(new_input_ids)

        if len(new_image_tensors) == 0:
            return None

        num_image_tokens = (new_input_ids == IMAGE_TOKEN_INDEX).sum().item()
        num_images = len(new_image_tensors)

        if num_images == 0 or num_image_tokens == 0:
            return None

        if num_image_tokens != num_images:
            print(f"Warning: Image token count mismatch - tokens: {num_image_tokens}, images: {num_images}")
            if num_images > num_image_tokens:
                new_image_tensors = new_image_tensors[:num_image_tokens]
            else:
                return None

        try:
            numeric_lengths = [item if isinstance(item, int) else item["length"] for item in new_length]
            token_diff = sum(numeric_lengths) - new_input_ids.shape[-1]
            expected_diff = len(new_length) * (NUM_IMAGES_TOKEN - 1)
            if token_diff != expected_diff:
                print(f"Warning: Length difference mismatch - Got {token_diff}, Expected {expected_diff}")
        except Exception as e:
            print(f"Warning: Error calculating length differences: {e}")

        if new_input_ids.shape[-1] > self.context_len:
            new_input_ids = new_input_ids[:self.context_len]
        elif new_input_ids.shape[-1] < self.context_len:
            padding = torch.full((self.context_len - new_input_ids.shape[-1],),
                                 self.pad_token_id,
                                 dtype=new_input_ids.dtype)
            new_input_ids = torch.cat([new_input_ids, padding])

        return {"image_tensors": new_image_tensors, "input_ids": new_input_ids, "lengths": new_length}

    def get_dataloader(self, batch_size=1):
        """Return a DataInfo with WebLoader for distributed training"""
        print(f"Creating WebLoader with batch_size={batch_size}, num_workers={self.num_workers}")

        loader = wds.WebLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.collator,
        )

        loader.num_batches = math.ceil(self.num_packs / batch_size)
        print(f"Dataloader configured with {loader.num_batches:,} batches per epoch")
        print(f"With gradient accumulation steps={8}, estimated training steps ~{loader.num_batches//8:,}")

        return DataInfo(dataloader=loader, shared_epoch=self.shared_epoch)