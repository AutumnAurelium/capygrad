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
        # """
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