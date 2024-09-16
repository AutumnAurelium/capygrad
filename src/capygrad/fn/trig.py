"""
Module for the trigonometric and hyperbolic trigonometric functions.
"""

from typing import List, Tuple

import numpy as np

from capygrad.fn.core import Function

class Sin(Function):
    """
    Sine function applied element-wise to the input tensor.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 1, "Sin only takes one input."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.sin(inputs[0])
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad * np.cos(self.inputs[0])]
    
    def name(self) -> str:
        return "Sin"

class Cos(Function):
    """
    Cosine function applied element-wise to the input tensor.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 1, "Cos only takes one input."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.cos(inputs[0])
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [-grad * np.sin(self.inputs[0])]
    
    def name(self) -> str:
        return "Cos"

class Tan(Function):
    """
    Tangent function applied element-wise to the input tensor.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 1, "Tan only takes one input."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.tan(inputs[0])
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad / (np.cos(self.inputs[0]) ** 2)]
    
    def name(self) -> str:
        return "Tan"

class Sinh(Function):
    """
    Hyperbolic sine function applied element-wise to the input tensor.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 1, "Sinh only takes one input."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.sinh(inputs[0])
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad * np.cosh(self.inputs[0])]
    
    def name(self) -> str:
        return "Sinh"

class Cosh(Function):
    """
    Hyperbolic cosine function applied element-wise to the input tensor.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 1, "Cosh only takes one input."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.cosh(inputs[0])
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad * np.sinh(self.inputs[0])]
    
    def name(self) -> str:
        return "Cosh"

class Tanh(Function):
    """
    Hyperbolic tangent function applied element-wise to the input tensor.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 1, "Tanh only takes one input."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.tanh(inputs[0])
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad * (1 - np.tanh(self.inputs[0])**2)]
    
    def name(self) -> str:
        return "Tanh"