"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional
import json
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling, PaddedCollatorForMMLanguageModeling, DataInfo

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-Device Batch Size must evenly divide Global Batch Size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int, save_vision_backbone: bool) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset,  # Can be Dataset or WDSPackedDataset
        collator,#: Optional[PaddedCollatorForLanguageModeling, PaddedCollatorForMMLanguageModeling],
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""

        # Load the latest checkpoint if available
        resume_step = self.load_latest_checkpoint(metrics.run_dir) or 0
        
        # Special handling for WebDataset streaming from S3
        if stage == "dynamic-pretrain":
            # Use WDSPackedDataset's built-in dataloader
            data_info = dataset.get_dataloader(batch_size=self.per_device_batch_size)
            return self.run_training_with_wds_dataloader(data_info, metrics, stage, seed)
        
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=8,
            worker_init_fn=self.worker_init_fn,
        )
        
        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            initial=resume_step,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # Restart from the resume step
                    if metrics.global_step < resume_step:
                        metrics.commit(global_step=metrics.global_step + 1)
                        continue
                    
                    try:
                        # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                        with torch.autocast(
                            "cuda",
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training,
                        ):
                            output: CausalLMOutputWithPast = self.vlm(
                                input_ids=batch["input_ids"],
                                attention_mask=getattr(batch, "attention_mask", None),
                                pixel_values=batch["pixel_values"],
                                labels=batch["labels"],
                                # multimodal_indices=batch["multimodal_indices"],
                            )
                            loss = output.loss

                        # Commit Loss (Prior to Gradient Accumulation Normalization)
                        metrics.commit(loss=loss)

                        # Normalize Loss to account for Gradient Accumulation --> Backward!
                        # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                        #             because in general, each batch has a *different number of masked out tokens* (because
                        #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                        #
                        #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                        #             the "correct" implementation, without adding extra complexity.
                        #
                        # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                        #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                        #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                        #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                        normalized_loss = loss / self.grad_accumulation_steps
                        normalized_loss.backward()

                        # Step =>> Only if Done w/ Gradient Accumulation
                        if (train_idx + 1) % self.grad_accumulation_steps == 0:
                            metrics.commit(update_step_time=True)

                            # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                            self.clip_grad_norm()

                            # Optimizer & LR Scheduler Step
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()

                            # Push Metrics
                            metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                            status = metrics.push()

                            # Save ckpt every 5k steps
                            # if self.max_steps is not None and metrics.global_step % 10000 == 0 and metrics.global_step < self.max_steps:
                            if metrics.global_step % 1000 == 0:
                                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                                dist.barrier()
                            
                            # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                            if self.max_steps is not None and metrics.global_step >= self.max_steps:
                                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                                dist.barrier()

                                return

                            # Update Progress Bar
                            progress.update()
                            progress.set_description(status)
                    except json.JSONDecodeError as ex:
                        # Log warning and continue to next batch
                        import logging
                        logging.warning(f"Skipping batch {train_idx} due to JSONDecodeError: {ex}")
                        continue

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    def run_training_with_wds_dataloader(
        self,
        data_info,
        metrics: Metrics,
        stage: str = "dynamic-pretrain",
        seed: int = 7,
    ) -> None:
        """Run training using a WebDataset dataloader for a single epoch."""
        # Extract loader and shared epoch info
        loader = data_info.dataloader
        # Load the latest checkpoint if available
        resume_step = self.load_latest_checkpoint(metrics.run_dir) or 0
        
        # For single-epoch training, ensure epochs is 1
        if self.epochs > 1 and overwatch.is_rank_zero():
            overwatch.info(f"Note: Multiple epochs ({self.epochs}) configured, but typically running for just one epoch")
        
        # Max Steps vs. Epochs Computation
        estimated_steps_per_epoch = loader.num_batches // self.grad_accumulation_steps
        
        # For single-epoch training with max_steps, ensure it doesn't exceed the epoch
        if self.max_steps is not None:
            if self.max_steps > estimated_steps_per_epoch and overwatch.is_rank_zero():
                overwatch.info(f"Warning: max_steps ({self.max_steps}) exceeds estimated steps in one epoch ({estimated_steps_per_epoch})")
                overwatch.info(f"Since you're running for one epoch, actual steps will be limited to {estimated_steps_per_epoch}")
        
        # Calculate the total number of steps for tqdm - for single epoch, this is just the steps in one epoch
        total_steps = self.max_steps if self.max_steps is not None else estimated_steps_per_epoch
        
        # Log training plan for clarity
        if overwatch.is_rank_zero():
            overwatch.info(f"=== Single-Epoch Training Plan ===")
            overwatch.info(f"Batches in epoch (estimated): {loader.num_batches:,}")
            overwatch.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")
            overwatch.info(f"Training steps (estimated): {estimated_steps_per_epoch:,}")
            if self.max_steps is not None:
                overwatch.info(f"Max steps override: {min(self.max_steps, estimated_steps_per_epoch):,}")
            overwatch.info(f"==================")
        
        # Keep track of actual batches processed
        actual_batches_processed = 0
        
        # Calculate checkpoint frequency - save approximately every 20-30 minutes of training
        # For expected ~500 steps, this would be around every 100 steps
        checkpoint_frequency = 500 # TODO (haoli): make this configurable via args
        if overwatch.is_rank_zero():
            overwatch.info(f"Will save checkpoints every {checkpoint_frequency} steps")

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=total_steps,
            initial=resume_step,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            # Single epoch training
            epoch = 0
            # Reseed WebDataset shuffle
            data_info.set_epoch(epoch)
            self.vlm.train()
            
            # Reset batch counter
            actual_batches_processed = 0
            epoch_start_time = time.time()
            
            # Periodic logging frequency - more frequent for single-epoch case
            log_frequency = min(50, max(1, estimated_steps_per_epoch // 20))  # Log ~20 times during training
            
            # Zero-Gradients (just in case)
            self.optimizer.zero_grad()
            
            # Track last active time to detect hangs
            last_activity_time = time.time()
            # Timeout detection - if no progress for 30 minutes, save a checkpoint
            activity_timeout_seconds = 1800  # 30 minutes

            # Set up special exception handling
            try:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                for train_idx, batch in enumerate(loader):
                    # Update activity time
                    last_activity_time = time.time()
                    
                    # Count this batch
                    actual_batches_processed += 1
                    
                    # Process the batch - convert to dict format if it's a tuple
                    if isinstance(batch, tuple):
                        batch_dict = {
                            "pixel_values": batch[0], 
                            "input_ids": batch[1],
                            "labels": batch[2]
                        }
                        batch = batch_dict
                    
                    # Restart from the resume step
                    if metrics.global_step < resume_step:
                        metrics.commit(global_step=metrics.global_step + 1)
                        continue
                    
                    try:
                        # Forward pass with autocast
                        with torch.autocast(
                            "cuda",
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training,
                        ):
                            output = self.vlm(
                                input_ids=batch["input_ids"],
                                attention_mask=getattr(batch, "attention_mask", None),
                                pixel_values=batch["pixel_values"],
                                labels=batch["labels"],
                            )
                            loss = output.loss

                        # Commit Loss (Prior to Gradient Accumulation Normalization)
                        metrics.commit(loss=loss)

                        # Normalize Loss and backward
                        normalized_loss = loss / self.grad_accumulation_steps
                        normalized_loss.backward()

                        # Step =>> Only if Done w/ Gradient Accumulation
                        if (train_idx + 1) % self.grad_accumulation_steps == 0:
                            metrics.commit(update_step_time=True)

                            # Clip Gradients
                            self.clip_grad_norm()

                            # Optimizer & LR Scheduler Step
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()

                            # Update step count and metrics
                            metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                            status = metrics.push()
                            
                            # Track current step progress
                            current_steps_completed = (actual_batches_processed // self.grad_accumulation_steps)
                            
                            # For single-epoch: If we've exceeded the estimated steps, we need to update the progress bar's total
                            if current_steps_completed > estimated_steps_per_epoch and self.max_steps is None:
                                # Only log this once when we first exceed the estimate
                                if current_steps_completed == estimated_steps_per_epoch + 1 and overwatch.is_rank_zero():
                                    overwatch.info(
                                        f"Actual steps ({current_steps_completed}) exceeds estimated steps ({estimated_steps_per_epoch}). "
                                        f"Adjusting progress bar."
                                    )
                                
                                # Dynamically update the progress bar total as we go
                                # For single-epoch, we can reasonably guess we're about x% through based on the
                                # current step vs. total samples
                                progress_fraction = min(0.99, actual_batches_processed / loader.num_batches)
                                if progress_fraction > 0:
                                    estimated_total_steps = int(current_steps_completed / progress_fraction)
                                    if abs(estimated_total_steps - progress.total) > max(10, progress.total * 0.05):  # Only update if significant change
                                        old_total = progress.total
                                        progress.total = estimated_total_steps
                                        if overwatch.is_rank_zero():
                                            overwatch.info(f"Adjusted progress bar total: {old_total:,} → {estimated_total_steps:,} steps")
                                        progress.refresh()

                            # More frequent checkpoint saving - ensure we don't lose progress
                            if metrics.global_step % checkpoint_frequency == 0:
                                if overwatch.is_rank_zero():
                                    overwatch.info(f"Saving periodic checkpoint at step {metrics.global_step}")
                                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                                dist.barrier()
                            
                            # Check for termination
                            if self.max_steps is not None and metrics.global_step >= self.max_steps:
                                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                                dist.barrier()
                                return

                            # Update Progress Bar
                            progress.update()
                            progress.set_description(status)
                            
                            # More frequent progress logging for single-epoch training
                            if metrics.global_step % log_frequency == 0 and overwatch.is_rank_zero():
                                elapsed_steps = metrics.global_step - resume_step
                                if elapsed_steps > 0:
                                    current_total = progress.total  # Use the potentially adjusted total
                                    percent_complete = (elapsed_steps / current_total) * 100 if current_total > 0 else 0
                                    remaining_steps = max(0, current_total - elapsed_steps)
                                    if current_steps_completed > 0:
                                        steps_per_sec = elapsed_steps / (time.time() - epoch_start_time)
                                        est_remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                                        overwatch.info(f"Step {metrics.global_step}: {percent_complete:.1f}% complete ({elapsed_steps}/{current_total} steps) "
                                                      f"- Est. remaining: {est_remaining_time/60:.1f} min")
                                    else:
                                        overwatch.info(f"Step {metrics.global_step}: {percent_complete:.1f}% complete ({elapsed_steps}/{current_total} steps)")
                                
                    except Exception as ex:
                        # More comprehensive error handling
                        import logging
                        if "CUDA out of memory" in str(ex):
                            logging.error(f"CUDA OOM error at batch {train_idx}: {ex}")
                            # Try to save checkpoint before crashing
                            try:
                                if overwatch.is_rank_zero():
                                    overwatch.info("Saving emergency checkpoint due to CUDA OOM")
                                emergency_path = metrics.run_dir / f"emergency_cuda_oom_step{metrics.global_step}"
                                self.save_checkpoint(emergency_path, metrics.global_step, epoch, loss.item() if 'loss' in locals() else None)
                            except:
                                logging.error("Could not save emergency checkpoint")
                            raise  # Re-raise to stop training
                        elif "JSONDecodeError" in str(ex) or "connection" in str(ex).lower() or "timeout" in str(ex).lower():
                            # Network/data errors - can continue
                            logging.warning(f"Recoverable error at batch {train_idx}, skipping: {ex}")
                            continue
                        else:
                            # Unknown error - log but try to continue
                            logging.error(f"Error at batch {train_idx}: {ex}")
                            continue
                    
                    # Check for potential hangs - if no progress for 30 minutes, save checkpoint
                    if time.time() - last_activity_time > activity_timeout_seconds:
                        if overwatch.is_rank_zero():
                            overwatch.info(f"No activity detected for {activity_timeout_seconds/60:.1f} minutes, saving emergency checkpoint")
                        emergency_path = metrics.run_dir / f"emergency_timeout_step{metrics.global_step}"
                        self.save_checkpoint(emergency_path, metrics.global_step, epoch, loss.item() if 'loss' in locals() else None)
                        last_activity_time = time.time()  # Reset timer
            
            except Exception as global_ex:
                # Catch any exceptions that might happen outside the batch loop
                if overwatch.is_rank_zero():
                    overwatch.info(f"Caught exception outside batch loop: {global_ex}")
                    overwatch.info("Saving emergency checkpoint")
                emergency_path = metrics.run_dir / f"emergency_global_step{metrics.global_step}"
                self.save_checkpoint(emergency_path, metrics.global_step, epoch, loss.item() if 'loss' in locals() else None)
                raise  # Re-raise after saving
            
            # Log the full training summary
            if overwatch.is_rank_zero():
                epoch_time = time.time() - epoch_start_time
                actual_steps = actual_batches_processed // self.grad_accumulation_steps
                overwatch.info(f"=== Training Complete ===")
                overwatch.info(f"Processed {actual_batches_processed:,} batches, {actual_steps:,} steps in {epoch_time:.1f}s")
                overwatch.info(f"Throughput: {actual_batches_processed/epoch_time:.1f} batches/sec, {actual_steps/epoch_time:.1f} steps/sec")
                
                # Report accuracy of initial estimate
                if abs(actual_steps - estimated_steps_per_epoch) > estimated_steps_per_epoch * 0.05:  # >5% difference
                    percent_diff = abs(actual_steps - estimated_steps_per_epoch)/estimated_steps_per_epoch * 100
                    overwatch.info(f"Initial step estimate was off by {percent_diff:.1f}%: estimated {estimated_steps_per_epoch:,}, actual {actual_steps:,}")

            # Save checkpoint at end of training
            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
            dist.barrier()

    def run_training_with_dataloader(
        self,
        datainfo: DataInfo,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        # if "finetune" in stage and batch_construction_strategy == "split-modality":
        #     # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
        #     #   (e.g., grouping by length) =>> can easily add them here!
        #     modality_lengths = dataset.get_modality_lengths()
        #     sampler = SplitModalitySampler(
        #         dataset,
        #         modality_lengths,
        #         global_batch_size=self.global_batch_size,
        #         num_replicas=overwatch.world_size(),
        #         rank=overwatch.rank(),
        #         seed=seed,
        #         drop_last=False,
        #     )

        # else:
        #     sampler = DistributedSampler(
        #         dataset,
        #         num_replicas=overwatch.world_size(),
        #         rank=overwatch.rank(),
        #         shuffle=True,
        #         seed=seed,
        #         drop_last=False,
        #     )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=self.per_device_batch_size,
        #     sampler=sampler,
        #     collate_fn=collator,
        #     num_workers=2,
        #     worker_init_fn=self.worker_init_fn,
        # )
        total_batches = datainfo.dataloader.num_batches
        
        # Max Steps vs. Epochs Computation
        steps_per_epoch = total_batches // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (total_batches // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                datainfo.set_epoch(epoch)
                dataloader = datainfo.dataloader
                self.vlm.train()

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    try:
                        # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                        with torch.autocast(
                            "cuda",
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training,
                        ):
                            pixel_values, input_ids, labels, attention_mask = batch

                            output: CausalLMOutputWithPast = self.vlm(
                                input_ids=input_ids,
                                pixel_values=pixel_values,
                                labels=labels,
                                attention_mask=attention_mask
                            )
                            # output: CausalLMOutputWithPast = self.vlm(
                            #     input_ids=batch["input_ids"],
                            #     attention_mask=batch["attention_mask"],
                            #     pixel_values=batch["pixel_values"],
                            #     labels=batch["labels"],
                            #     multimodal_indices=batch["multimodal_indices"],
                            # )
                            loss = output.loss

                        # Commit Loss (Prior to Gradient Accumulation Normalization)
                        metrics.commit(loss=loss)

                        # Normalize Loss to account for Gradient Accumulation --> Backward!
                        # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                        #             because in general, each batch has a *different number of masked out tokens* (because
                        #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                        #
                        #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                        #             the "correct" implementation, without adding extra complexity.
                        #
                        # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                        #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                        #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                        #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                        normalized_loss = loss / self.grad_accumulation_steps
                        normalized_loss.backward()

                        # Step =>> Only if Done w/ Gradient Accumulation
                        if (train_idx + 1) % self.grad_accumulation_steps == 0:
                            metrics.commit(update_step_time=True)

                            # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                            self.clip_grad_norm()

                            # Optimizer & LR Scheduler Step
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()

                            # Push Metrics
                            metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                            status = metrics.push()

                            # Save ckpt every 1k steps
                            if self.max_steps is not None and metrics.global_step % 1000 == 0 and metrics.global_step < self.max_steps:
                                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                                dist.barrier()

                            # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                            if self.max_steps is not None and metrics.global_step >= self.max_steps:
                                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                                dist.barrier()

                                return

                            # Update Progress Bar
                            progress.update()
                            progress.set_description(status)
                    except json.JSONDecodeError as ex:
                        # Log warning and continue to next batch
                        import logging
                        logging.warning(f"Skipping batch {train_idx} due to JSONDecodeError: {ex}")
                        continue

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()