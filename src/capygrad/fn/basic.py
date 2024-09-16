from typing import List, Tuple

import numpy as np

from capygrad.fn.core import Function

class Add(Function):
    """
    Simple addition. NumPy's semantics allow you to add a scalar to a tensor, but this causes unintuitive behavior in the backward pass, so it is not permitted here.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 2, "Add only takes two inputs."
        assert input_shapes[0] == input_shapes[1], "Shapes must be the same."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return inputs[0] + inputs[1]
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad, grad]
    
    def name(self) -> str:
        return "Add"

class ElemMul(Function):
    """
    Element-wise multiplication.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 2, "Mul only takes two inputs."
        assert (input_shapes[0] == input_shapes[1]) or (input_shapes[0] == (1,) or input_shapes[1] == (1,)), "Shapes must be the same or one of them must be (1,)."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return inputs[0] * inputs[1]
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad * self.inputs[1], grad * self.inputs[0]]
    
    def name(self) -> str:
        return "Mul"
    
class Pow(Function):
    """
    Power function.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 2, "Pow only takes two inputs."
        assert input_shapes[1] == (1,), "The second input must be a scalar."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return inputs[0] ** inputs[1].astype(np.float64)  # we need to do this, because numpy doesn't like integers to negative powers.
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        in0_float = self.inputs[0].astype(np.float64)
        return [grad * self.inputs[1] * (in0_float ** (self.inputs[1] - 1)), np.sum(grad * (in0_float ** self.inputs[1]) * np.log(in0_float))]
    
    def name(self) -> str:
        return "Pow"

class Div(Function):
    """
    Element-wise division.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 2, "Div only takes two inputs."
        assert (input_shapes[0] == input_shapes[1]) or (input_shapes[1] == (1,)), "Shapes must be the same or the second input must be a scalar."
        return input_shapes[0]
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return inputs[0] / inputs[1]
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        # Gradient for the numerator
        grad_0 = grad / self.inputs[1]
        
        # Gradient for the denominator
        grad_1 = -grad * self.inputs[0] / (self.inputs[1] ** 2)
        
        # If the denominator is a scalar, we need to sum the gradient
        if self.inputs[1].shape == (1,):
            grad_1 = np.sum(grad_1)
        
        return [grad_0, grad_1]
    
    def name(self) -> str:
        return "Div"

class MatMul(Function):
    """
    Standard matrix multiplication.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 2, "MatMul only takes two inputs."
        assert len(input_shapes[0]) == 2, "The first input must be a matrix."
        assert len(input_shapes[1]) == 2, "The second input must be a matrix."
        assert input_shapes[0][1] == input_shapes[1][0], "The number of columns in the first input must be equal to the number of rows in the second input."
        return (input_shapes[0][0], input_shapes[1][1])
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return inputs[0] @ inputs[1]
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad @ self.inputs[1].T, self.inputs[0].T @ grad]
    
    def name(self) -> str:
        return "MatMul"
    
class Sum(Function):
    """
    Sums every element in a vector.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        assert len(input_shapes) == 1, "Sum only takes one input."
        return (1,)
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        self.inputs = inputs
        return np.sum(inputs[0])
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad * np.ones_like(self.inputs[0])]
    
    def name(self) -> str:
        return "Sum"