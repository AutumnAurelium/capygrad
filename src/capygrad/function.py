from typing import List, Tuple

import numpy as np


class Function:
    """
    A function instance. This is a base class for all functions.
    A single instance of this class is created for each Value that uses it. The `self` object passed to forward() should be used to store information for the backward pass.
    """
    def shape(self, input_shapes: List[Tuple]) -> Tuple[int, ...]:
        """
        Returns a numpy shape tuple for the output tensor given all input tensors.
        It is acceptable to make assertions in this function to validate input shapes, such as:
        
        ```python
        assert input_shape == (4, 1)
        ```
        """
        raise NotImplementedError()
    
    def forward(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        Performs the forward pass. Given a list of inputs, returns the output tensor. You may assume that shape() has been called and any asserts in it have passed.
        
        This function should store all necessary inputs and intermediate results for the backward pass in `self`.
        """
        raise NotImplementedError()
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        """
        Performs the backward pass. Given the gradient of the output tensor, returns the gradients of the input tensors.
        """
        raise NotImplementedError()
    
    def name(self) -> str:
        """
        Returns the name of this function, for debugging and visualization purposes.
        """
        raise NotImplementedError()

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
        return inputs[0] ** inputs[1]
    
    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        return [grad * self.inputs[1] * (self.inputs[0] ** (self.inputs[1] - 1)), np.sum(grad * (self.inputs[0] ** self.inputs[1]) * np.log(self.inputs[0]))]
    
    def name(self) -> str:
        return "Pow"

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