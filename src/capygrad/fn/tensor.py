"""
Module for functions related to manipulating tensors (stacking, splitting, etc.)
"""

from typing import List, Tuple

import numpy as np

from capygrad.fn.core import Function

class Stack(Function):
    """
    Combines some number of N-dimensional tensors into one (N+1)-dimensional tensors, arranged in the order they're passed in.
    
    For example, 8 10x10 matrices can be stacked into a 8x10x10 tensor.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int]:
        assert np.all(np.equal(input_shapes, input_shapes[0])), "All inputs to Combine must have the same dimensions."
        return (len(input_shapes),) + input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.stack(inputs)
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad[i] for i in range(len(self.inputs))]
    
    def name(self) -> str:
        return "Stack"