# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
from functools import partial
from pathlib import Path
from torch.utils.data import Sampler
import math

import torch
import random

import logging
logger = logging.getLogger(__name__)


import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class DistributedDynamicBatchSampler(Sampler):
    def __init__(self, dataset, max_frames_per_batch, num_replicas=None, rank=None, shuffle=True, seed=0):

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.max_frames = max_frames_per_batch
        self.shuffle = shuffle
        self.seed = seed

        # --- ROBUST INITIALIZATION ---
        # Standard DDP Initialization
        # 1. If num_replicas/rank are NOT provided, try to get them from dist
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                # Fallback: If DDP isn't init'd yet, assume single GPU to avoid crash
                num_replicas = 1 
                
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0 # Fallback
        
        self.num_replicas = num_replicas
        self.rank = rank


    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # --- STEP 1: YOUR ORIGINAL LOGIC (Global) ---
        # We must do this globally first so length-grouping works across all GPUs
        
        indices = list(range(len(self.dataset)))
        # Sort by length (using your specific accessor)
        indices.sort(key=lambda i: self.dataset.get_encoded_len(self.dataset.data_list[i]), reverse=False)

        # Form batches (Your original logic, just wrapped here)
        all_batches = []
        batch = []
        batch_frames = 0
        # import pdb; pdb.set_trace()
        for idx in indices:
            item = self.dataset.data_list[idx]
            n_frames = int(item["encoder_len"])

            if len(batch) > 0 and (batch_frames + n_frames) > self.max_frames:
                all_batches.append(batch)
                batch = []
                batch_frames = 0
            
            if n_frames < self.max_frames:
                batch.append(idx)
                batch_frames += n_frames
        
        if len(batch) > 0:
            all_batches.append(batch)

        # --- STEP 2: STANDARD DDP SHUFFLING ---
        # Deterministically shuffle the LIST of batches (not the items inside)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # Shuffle the order of batches
            shuffle_ids = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in shuffle_ids]

        # --- STEP 3: SUBSAMPLE FOR THIS SPECIFIC GPU ---
        # This is the "Distributed" part. 
        # GPU 0 gets indices [0, 4, 8...], GPU 1 gets [1, 5, 9...]
        my_batches = all_batches[self.rank :: self.num_replicas]

        for batch in my_batches:
            yield batch

    def __len__(self):
        # We can't know the exact number without running the loop, 
        # but this approximation is safe for progress bars.
        total_frames = sum(self.dataset.get_encoded_len(x) for x in self.dataset.data_list)
        est_total_batches = total_frames / self.max_frames
        return math.ceil(est_total_batches / self.num_replicas)



def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str, num_lang: int, **kwargs):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split, num_lang, **kwargs)
    except AttributeError as e:
        logger.info(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train", num_lang=1, **kwargs
) -> torch.utils.data.Dataset:

    return get_custom_dataset(
        dataset_config,
        tokenizer,
        split,
        num_lang,
        **kwargs
    )
