"""
Key features of Perception Memory:
1. Efficient Caching System
2. Memory Optimization
3. Robust Tagging System
4. Performance Features
5. Diagnostic Capabilities
6. Safety Mechanisms
"""
import os
import torch
import hashlib
import warnings
import json, yaml

from torch import nn, einsum
from collections import OrderedDict, defaultdict
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict, Any, Union, List, Callable

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Perception Memory")
printer = PrettyPrinter

class PerceptionMemory(nn.Module):
    """
    Perception Memory prioritizes memory efficiency while maintaining flexibility for different use cases.
    The combination of in-memory caching, gradient checkpointing,
    and disk storage provides a comprehensive memory management solution suitable for large-scale perception systems.
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.memory_config = get_config_section('perception_memory')
        self.cache = OrderedDict()
        self.tag_index = defaultdict(list)
        self.checkpoint_dir = self.memory_config.get('checkpoint_dir')
        self.cache_dir = self.memory_config.get('cache_dir')
        self.max_cache_size = self.memory_config.get('max_cache_size')
        self.enable_checkpointing = self.memory_config.get('enable_checkpointing')
        self.enable_cache = self.memory_config.get('enable_cache')

        if self.enable_checkpointing and self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.enable_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Memory metrics
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.total_stored = 0

        # Register cleanup hook
        #self.register_buffer('dummy', torch.tensor(0))
        self._register_cleanup_hook()

    def _register_cleanup_hook(self):
        """Register hook to clean up when module is deleted"""
        def cleanup_hook(module, inputs):
            self.clear_cache()
            logger.info("Perception Memory cache cleared during destruction")

        self.register_forward_pre_hook(cleanup_hook)

    def cache_item(self, 
                 tensor: torch.Tensor, 
                 key: Optional[str] = None, 
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None) -> str:
        """Cache a tensor with optional tags and metadata"""
        printer.status("MEMORY", "Caching items", "info")

        if not self.enable_cache:
            return ""

        # Generate key if not provided
        if key is None:
            key = self._generate_key(tensor.shape, tensor.dtype, tensor.device)

        # Manage cache size with LRU eviction
        if len(self.cache) >= self.max_cache_size:
            evicted_key, _ = self.cache.popitem(last=False)
            self._remove_from_tag_index(evicted_key)
            self.eviction_count += 1
            logger.debug(f"Evicted cache item: {evicted_key}")

        # Detach tensor to save memory
        cached_tensor = tensor.detach().cpu()

        # Store in cache
        self.cache[key] = {
            'tensor': cached_tensor,
            'tags': tags or [],
            'metadata': metadata or {},
            'access_count': 0
        }

        # Update tag index
        if tags:
            for tag in tags:
                self.tag_index[tag].append(key)

        self.total_stored += 1
        return key

    def _generate_key(self, *args) -> str:
        """Generate unique key from input arguments"""
        printer.status("MEMORY", "Generate unique key", "info")

        key_str = "|".join(str(arg) for arg in args)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _remove_from_tag_index(self, key: str):
        """Remove key from tag index"""
        printer.status("MEMORY", "Removing key from tag index", "info")

        for tag, keys in self.tag_index.items():
            if key in keys:
                keys.remove(key)
            if not keys:
                del self.tag_index[tag]

    def retrieve(self, 
               key: Optional[str] = None, 
               tag: Optional[str] = None,
               device: Optional[torch.device] = None) -> torch.Tensor:
        """Retrieve tensor by key or tag"""
        printer.status("MEMORY", "Retrieving tensor by key or tag", "info")

        if not self.enable_cache:
            raise RuntimeError("Cache is disabled")
            
        # Retrieve by key
        if key is not None:
            if key in self.cache:
                self.cache.move_to_end(key)  # Mark as recently used
                self.cache[key]['access_count'] += 1
                self.hit_count += 1
                tensor = self.cache[key]['tensor']
                return tensor.to(device) if device else tensor
            self.miss_count += 1
            raise KeyError(f"Key not found in cache: {key}")
            
        # Retrieve by tag
        if tag is not None:
            if tag in self.tag_index:
                keys = self.tag_index[tag]
                results = []
                for key in keys:
                    self.cache.move_to_end(key)
                    self.cache[key]['access_count'] += 1
                    tensor = self.cache[key]['tensor']
                    results.append(tensor.to(device) if device else tensor)
                self.hit_count += len(results)
                return results
            self.miss_count += 1
            raise KeyError(f"Tag not found in cache: {tag}")
            
        raise ValueError("Either key or tag must be provided")

    def clear_cache(self, key: Optional[str] = None, tag: Optional[str] = None):
        """Clear cache by key, tag, or entirely"""
        printer.status("MEMORY", "Clearing cache by key, tag, or entirely", "info")

        if key:
            if key in self.cache:
                del self.cache[key]
                self._remove_from_tag_index(key)
                logger.debug(f"Cleared cache item: {key}")
            return
            
        if tag:
            if tag in self.tag_index:
                keys = self.tag_index.pop(tag)
                for key in keys:
                    if key in self.cache:
                        del self.cache[key]
                logger.debug(f"Cleared {len(keys)} items with tag: {tag}")
            return
            
        # Clear entire cache
        self.cache.clear()
        self.tag_index.clear()
        logger.info("PerceptionMemory cache cleared")

    def run_checkpointed(self, 
                       fn: Callable, 
                       *args, 
                       preserve_rng_state: bool = False,
                       **kwargs) -> Any:
        """Run function with gradient checkpointing"""
        printer.status("MEMORY", "Running function with gradient checkpointing", "info")

        if not self.enable_checkpointing:
            return fn(*args, **kwargs)
            
        return checkpoint(
            fn, 
            *args, 
            preserve_rng_state=preserve_rng_state,
            **kwargs
        )

    def checkpoint(self, 
                 tensor: torch.Tensor, 
                 file_prefix: str = "memory_checkpoint",
                 metadata: Optional[Dict] = None) -> str:
        """Save tensor to disk checkpoint"""
        printer.status("MEMORY", "Saving tensor to disk checkpoint", "info")

        metadata = metadata or {}
        metadata.update({
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device)
        })
        
        # Create filename
        file_hash = hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
        filename = f"{self.checkpoint_dir}/{file_prefix}_{file_hash}.pt"
        
        # Save tensor and metadata
        torch.save({
            'tensor': tensor.cpu(),
            'metadata': metadata
        }, filename)
        
        logger.debug(f"Checkpoint saved to {filename}")
        return filename

    def load_checkpoint(self, filename: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Load tensor from disk checkpoint"""
        printer.status("MEMORY", "Loading checkpoint...", "info")

        checkpoint = torch.load(filename, map_location=device)
        tensor = checkpoint['tensor']
        logger.debug(f"Loaded checkpoint from {filename}")
        return tensor.to(device) if device else tensor

    def memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        printer.status("MEMORY", "Retrieving memory stats", "info")

        cache_size = sum(
            t['tensor'].element_size() * t['tensor'].numel() 
            for t in self.cache.values()
        ) / (1024 ** 2)  # in MB
        
        return {
            'cache_size': len(self.cache),
            'memory_usage_mb': cache_size,
            'hit_rate': self.hit_count / (self.hit_count + self.miss_count) 
                        if (self.hit_count + self.miss_count) > 0 else 0,
            'total_stored': self.total_stored,
            'eviction_count': self.eviction_count,
            'enable_cache': self.enable_cache,
            'enable_checkpointing': self.enable_checkpointing
        }

    def toggle_cache(self, enable: bool):
        """Enable or disable caching"""
        self.enable_cache = enable
        logger.info(f"Caching {'enabled' if enable else 'disabled'}")

    def toggle_checkpointing(self, enable: bool):
        """Enable or disable gradient checkpointing"""
        self.enable_checkpointing = enable
        logger.info(f"Gradient checkpointing {'enabled' if enable else 'disabled'}")

    def forward(self, 
                input: Optional[Union[torch.Tensor, str, List[str]]] = None,
                operation: str = 'auto',
                tags: Optional[List[str]] = None,
                metadata: Optional[Dict] = None,
                device: Optional[torch.device] = None,
                file_prefix: str = "memory_forward",
                preserve_rng_state: bool = False,
                checkpoint_fn: Optional[Callable] = None) -> Union[torch.Tensor, str, List[torch.Tensor], None]:
        """
        Unified forward method for memory operations with intelligent auto-detection.
        
        Args:
            input: Can be:
                - Tensor (for caching)
                - Key string (for retrieval)
                - List of tags (for tag-based retrieval)
            operation: One of ['auto', 'cache', 'retrieve', 'retrieve_by_tag', 'checkpoint', 'run_checkpointed']
            tags: List of tags for caching or retrieval
            metadata: Additional metadata for caching
            device: Target device for retrieved tensors
            file_prefix: Prefix for checkpoint files
            preserve_rng_state: For gradient checkpointing
            checkpoint_fn: Function to run with gradient checkpointing
            
        Returns:
            Depending on operation:
                - Cache: Cache key (str)
                - Retrieve: Tensor or list of tensors
                - Checkpoint: Checkpoint filename (str)
                - Run_checkpointed: Result of checkpointed function
        """
        printer.status("MEMORY", f"PerceptionMemory forward - Operation: {operation}", "info")
        
        # Auto-detect operation based on input type
        if operation == 'auto':
            if isinstance(input, torch.Tensor):
                operation = 'cache'
            elif isinstance(input, str):
                operation = 'retrieve'
            elif isinstance(input, list) and all(isinstance(i, str) for i in input):
                operation = 'retrieve_by_tag'
            else:
                raise ValueError("Could not auto-detect operation from input type")
            printer.status("MEMORY", f"Auto-detected operation: {operation}", "debug")

        # Execute the requested operation
        if operation == 'cache':
            if not isinstance(input, torch.Tensor):
                raise TypeError("Input must be a tensor for caching operation")
            return self.cache_item(input, tags=tags, metadata=metadata)
        
        elif operation == 'retrieve':
            if not isinstance(input, str):
                raise TypeError("Input must be a string key for retrieval")
            return self.retrieve(key=input, device=device)
        
        elif operation == 'retrieve_by_tag':
            tags = input if isinstance(input, list) else [input]
            if not all(isinstance(t, str) for t in tags):
                raise TypeError("Tags must be strings for tag-based retrieval")
            
            # Handle multiple tags with intersection
            results = {}
            for tag in tags:
                items = self.retrieve(tag=tag, device=device)
                for i, item in enumerate(items):
                    # Use a composite key to track items across tags
                    item_key = f"{tag}_{i}"
                    if item_key not in results:
                        results[item_key] = {'tensor': item, 'count': 1}
                    else:
                        results[item_key]['count'] += 1
            
            # Return items that match all tags (intersection)
            matched_items = [v['tensor'] for v in results.values() if v['count'] == len(tags)]
            return matched_items
        
        elif operation == 'checkpoint':
            if not isinstance(input, torch.Tensor):
                raise TypeError("Input must be a tensor for checkpointing")
            return self.checkpoint(input, file_prefix=file_prefix, metadata=metadata)
        
        elif operation == 'run_checkpointed':
            if checkpoint_fn is None:
                raise ValueError("checkpoint_fn must be provided for run_checkpointed operation")
            return self.run_checkpointed(
                checkpoint_fn, 
                input, 
                preserve_rng_state=preserve_rng_state,
                **metadata if metadata else {}
            )
        
        elif operation == 'update':
            if not isinstance(input, tuple) or len(input) != 2:
                raise TypeError("Input must be a tuple (key, tensor) for update")
            key, tensor = input
            self.clear_cache(key=key)
            return self.cache_item(tensor, key=key, tags=tags, metadata=metadata)
        
        elif operation == 'stats':
            return self.memory_stats()
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
    def expensive_operation(x, y):
        # Complex computation
        result = {}
        return result

    def __repr__(self):
        stats = self.memory_stats()
        return (f"PerceptionMemory(cache_size={stats['cache_size']}, "
                f"memory={stats['memory_usage_mb']:.2f}MB, "
                f"hit_rate={stats['hit_rate']:.2f})")

if __name__ == "__main__":
    print("\n=== Running Perception Memory Test ===\n")
    printer.status("Init", "Perception Memory initialized", "success")
    memory = PerceptionMemory()

    print(memory)
    print("\n* * * * * Phase 2 - Extracted Features * * * * *\n")
    tensor = torch.randn(1, 128, 512)
    tags = ["feature", "test"]
    metadata = {"source": "unit_test", "shape": list(tensor.shape)}
    device = torch.device("cpu")

    key = memory.cache_item(tensor=tensor, tags=tags, metadata=metadata)
    retrieval = memory.retrieve(key=key, device=device)

    printer.pretty("CACHE", key, "success")
    printer.pretty("RETRIEVE", retrieval, "success")

    print("\n=== Successfully Ran Perception Memory ===\n")
