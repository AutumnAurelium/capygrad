from typing import List, Tuple, Union

import numpy as np

import capygrad.function as fn
from capygrad.function import Function

Literal = float | int | np.ndarray | list
ValueOrLiteral = Union["Value", Literal]

class Value:
    data: np.ndarray  # Cached data for this value.
    grad: np.ndarray  # Cached gradient for this value.
    
    needs_grad: bool  # Whether this value needs to have gradients computed for it.
    
    function: Function | None  # A function of None means that this is a variable.
    
    label: str | None  # A label for this value, for debugging or visualization purposes.
    
    parents: List["Value"]  # The value that this value was created from.
    
    def __init__(self, data: Literal | None, label=None, needs_grad=False, function: Function | None = None, parents: List["Value"] | None = None):
        if data is not None:
            self.data = np.array(data)
        elif parents:
            self.data = function.forward([x.data for x in parents])
        else:
            raise ValueError("Either data or function & parents must be provided.")

        self.grad = np.zeros_like(self.data)
        self.needs_grad = needs_grad
        self.label = label
        self.function = function
        self.parents = parents or []
    
    def __repr__(self):
        if self.label:
            return f"{self.label}={self.data}"
        elif self.function:
            return f"{self.function.name()}={self.data}"
        else:
            return f"{self.data}"
    
    def sum(self) -> "Value":
        return Value(function=fn.Sum(), parents=[self], needs_grad=self.needs_grad)
    
    def __add__(self, other: ValueOrLiteral) -> "Value":
        other = maybe_literal(other)
        return Value(function=fn.Add(), parents=[self, other], needs_grad=self.needs_grad or other.needs_grad)
    
    def __sub__(self, other: ValueOrLiteral) -> "Value":
        return self + -other

    def __neg__(self) -> "Value":
        return self * -1
    
    def __mul__(self, other: ValueOrLiteral) -> "Value":
        """
        This is element-wise multiplication.
        """
        other = maybe_literal(other)
        return Value(function=fn.ElemMul(), parents=[self, other], needs_grad=self.needs_grad or other.needs_grad)
    
    def __truediv__(self, other: ValueOrLiteral) -> "Value":
        other = maybe_literal(other)
        return self * other**-1
    
    def __pow__(self, other: ValueOrLiteral) -> "Value":
        other = maybe_literal(other)
        return Value(function=fn.Pow(), parents=[self, other], needs_grad=self.needs_grad or other.needs_grad)
    
    def __matmul__(self, other: ValueOrLiteral) -> "Value":
        other = maybe_literal(other)
        return Value(function=fn.MatMul(), parents=[self, other], needs_grad=self.needs_grad or other.needs_grad)
    
    def __radd__(self, other: ValueOrLiteral) -> "Value":
        return self + other
    
    def __rmul__(self, other: ValueOrLiteral) -> "Value":
        return self * other
    
    def __rtruediv__(self, other: ValueOrLiteral) -> "Value":
        return other / self
    
    def __rpow__(self, other: ValueOrLiteral) -> "Value":
        return other ** self
    
    def __rmatmul__(self, other: ValueOrLiteral) -> "Value":
        return other @ self

def maybe_literal(x: Literal | Value) -> Value:
    """
    If necessary, converts a literal (float, int, list, NP array) into a Value.
    Otherwise, returns the value as-is.
    """
    if isinstance(x, Value):
        return x
    else:
        return Value(np.array(x))