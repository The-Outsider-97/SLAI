import logging
import torch
import torch.nn as nn
from typing import Any, Union, Optional, Tuple, List

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("State Processor")

class StateProcessor(nn.Module):
    def __init__(self, env,
                 normalize: bool = False,
                 low: Optional[torch.Tensor] = None,
                 high: Optional[torch.Tensor] = None):
        """
        Process environment states into normalized tensors using PyTorch
        
        Args:
            normalize: Whether to normalize states using bounds
            low: Lower bounds for normalization
            high: Upper bounds for normalization
        """
        super().__init__()
        self.env = env
        self.normalize = normalize
        self.config = load_global_config()
        self.sp_config = get_config_section('state_processor_config')

        if normalize:
            if low is None or high is None:
                raise ValueError("Normalization requires low and high bounds")
            self.register_buffer('low', low.float())
            self.register_buffer('high', high.float())
            self.register_buffer('range_val', (high - low).float())
            self.range_val[self.range_val < 1e-6] = 1.0

    def process(self, state: Any) -> torch.Tensor:
        """
        Primary processing method required by LearningAgent
        Converts raw state into processed float tensor
        
        Args:
            state: Raw state from environment
            
        Returns:
            Processed float32 tensor
        """
        return self.forward(state)

    def forward(self, state: Any) -> torch.Tensor:
        """
        Convert raw state into processed float tensor
        
        Args:
            state: Raw state from environment. Can be:
                - Tuple (Gymnasium style)
                - List or tuple of numbers
                - Numpy array
                - Torch tensor
                - Scalar
                
        Returns:
            Flattened float32 tensor
        """
        if state is None:
            self.logger.warning("Received None state, returning empty tensor")
            return torch.tensor([], dtype=torch.float32)

        # Handle Gymnasium-style tuple (state, info)
        if isinstance(state, tuple) and len(state) >= 1:
            state = state[0]

        # Process different state types
        if isinstance(state, torch.Tensor):
            processed = state
        elif isinstance(state, (list, tuple)):
            processed = self._flatten_nested(state)
        elif hasattr(state, '__array__'):  # Handle numpy arrays
            processed = torch.from_numpy(state.__array__()).float()
        else:  # Handle scalars and other types
            processed = torch.tensor(state, dtype=torch.float32)

        # Ensure float32 and flatten
        processed = processed.to(torch.float32).flatten()

        # Apply normalization if configured
        if self.normalize:
            processed = self._normalize(processed)
            
        return processed

    def _flatten_nested(self, state: Union[list, tuple]) -> torch.Tensor:
        """
        Recursively flatten nested list/tuple structures using PyTorch
        
        Args:
            state: Nested list/tuple structure
            
        Returns:
            Flattened 1D tensor
        """
        flattened = []
        stack = [state]
        
        while stack:
            current = stack.pop()
            if isinstance(current, (list, tuple)):
                stack.extend(reversed(current))
            else:
                # Convert to tensor and handle numeric types
                try:
                    flattened.append(float(current))
                except TypeError:
                    self.logger.warning(f"Non-numeric element in state: {current}")
                    flattened.append(0.0)
                
        return torch.tensor(flattened, dtype=torch.float32)

    def _normalize(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state using configured bounds
        
        Args:
            state: Input state tensor
            
        Returns:
            Normalized state tensor in [0, 1] range
        """
        # Ensure tensors are on same device
        return (state - self.low) / self.range_val

    def update_bounds(self, 
                     low: torch.Tensor, 
                     high: torch.Tensor) -> None:
        """
        Update normalization bounds dynamically
        
        Args:
            low: New lower bounds
            high: New upper bounds
        """
        if not self.normalize:
            self.logger.warning("Updating bounds but normalization is disabled")
            return
            
        # Update buffers directly
        self.low.copy_(low.float())
        self.high.copy_(high.float())
        self.range_val.copy_((high - low).float())
        self.range_val[self.range_val < 1e-6] = 1.0
        self.logger.info("Normalization bounds updated")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running State Processor ===\n")
    # Initialize without normalization
    processor = StateProcessor()
    print(f"Processor created: {processor}")

    # Test processing different types
    print("\nTesting state processing:")
    tests = [
        ("Scalar", 5.0, torch.tensor([5.0])),
        ("List", [1, [2, 3], 4], torch.tensor([1.0, 2.0, 3.0, 4.0])),
        ("Tensor", torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([1.0, 2.0, 3.0, 4.0])),
        ("Gymnasium Tuple", (torch.tensor([0.5, 0.5]), {"info": "data"}), torch.tensor([0.5, 0.5]))
    ]
    
    for name, input_state, expected in tests:
        result = processor.process(input_state)
        match = torch.allclose(result, expected, atol=1e-4)
        print(f"- {name}: {'✓' if match else '✗'} {result.tolist()}")

    # Test normalization
    print("\n* * * * * Phase 2 * * * * *")
    print("== Testing normalization: ==\n")
    low = torch.tensor([-1.0, -2.0])
    high = torch.tensor([1.0, 2.0])
    norm_processor = StateProcessor(normalize=True, low=low, high=high)
    
    states = [
        ("Simple", torch.tensor([0.0, 0.0]), torch.tensor([0.5, 0.5])),
        ("Out-of-bound", torch.tensor([2.0, -3.0]), torch.tensor([1.5, -0.5]))
    ]
    
    for name, input_state, expected in states:
        result = norm_processor.process(input_state)
        match = torch.allclose(result, expected, atol=1e-4)
        print(f"- {name}: {'✓' if match else '✗'} {result.tolist()}")

    # Test bounds update
    print("\n* * * * * Phase 3 * * * * *")
    print("== Testing bounds update: ==\n")
    new_low = torch.tensor([-2.0, -4.0])
    new_high = torch.tensor([2.0, 4.0])
    norm_processor.update_bounds(new_low, new_high)
    
    input_state = torch.tensor([0.0, 0.0])
    expected = torch.tensor([0.5, 0.5])
    result = norm_processor.process(input_state)
    match = torch.allclose(result, expected, atol=1e-4)
    print(f"- Updated bounds: {'✓' if match else '✗'} {result.tolist()}")

    print("\n=== Successfully Ran State Processor ===")
