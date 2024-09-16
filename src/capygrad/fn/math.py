"""
A module for more complex mathematical operations.
"""

from typing import List, Tuple

import numpy as np

from capygrad.fn.core import Function

class Exp(Function):
    """
    Exponential function applied element-wise to the input tensor.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 1, "Exp only takes one input."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.exp(inputs[0])
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad * np.exp(self.inputs[0])]
    
    def name(self) -> str:
        return "Exp"
    
class Abs(Function):
    """
    Absolute value function applied element-wise to the input tensor.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 1, "Abs only takes one input."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.abs(inputs[0])
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad * np.sign(self.inputs[0])]
    
    def name(self) -> str:
        return "Abs"
