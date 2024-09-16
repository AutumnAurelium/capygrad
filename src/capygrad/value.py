from typing import List, Tuple, Union

import numpy as np

from capygrad.fn.basic import *
from capygrad.fn.core import Function
import capygrad.fn as fn

Literal = float | int | np.ndarray | list
ValueOrLiteral = Union["Value", Literal]

class Value:
    data: np.ndarray  # Cached data for this value.
    grad: np.ndarray  # Cached gradient for this value.
    
    needs_grad: bool  # Whether this value needs to have gradients computed for it.
    
    function: Function | None  # A function of None means that this is a variable.
    
    label: str | None  # A label for this value, for debugging or visualization purposes.
    
    parents: List["Value"]  # The value that this value was created from.
    
    def __init__(self, data: Literal | None = None, label=None, needs_grad=False, function: Function | None = None, parents: List[ValueOrLiteral] | None = None):
        self.needs_grad = needs_grad
        self.label = label
        self.function = function
        self.parents = [maybe_literal(x) for x in parents] if parents else []  # Allow us to use literals in the parents list.
        
        if data is not None:
            self.data = np.array(data)
        elif parents:
            self.data = function.forward([x.data for x in self.parents])
            
            # If any of the parents need gradients, then this value needs gradients.
            if True in [x.needs_grad for x in self.parents]:
                self.needs_grad = True
        else:
            raise ValueError("Either data or function & parents must be provided.")
        
        self.grad = np.zeros_like(self.data)
    
    def __repr__(self):
        if self.label:
            return f"{self.label}={self.data}"
        elif self.function:
            return f"{self.function.name()}={self.data}"
        else:
            return f"{self.data}"
    
    def forward(self):
        """
        Recursively recalculates the value of every Value upstream of this one, and then this Value.
        """
        for parent in self.parents:
            parent.forward()
        
        self.data = self.function.forward([x.data for x in self.parents])
        
    def backward(self):
        assert self.data.shape == (1) or self.data.shape == (), "You can only calculate gradients relative to a scalar."
        
        order = self.backward_ordering()
        
        # Zero out grads
        for val in order:
            val.grad = np.zeros_like(val.data, dtype=np.float64)
        
        # Set the gradient of this value to 1.0
        self.grad = np.ones_like(self.data, dtype=np.float64)
        
        # In traversal order, 1. compute the gradient of the current value, 2. add it to the gradient of the parents.
        for val in self.backward_ordering():
            if val.needs_grad:  # This would cause a bug if somehow a parent needs gradients but this value doesn't, but that shouldn't happen.
                if val.function is not None:
                    for i, grad in enumerate(val.function.backward(val.grad)):
                        val.parents[i].grad += grad
    
    def backward_ordering(self) -> List["Value"]:
        """
        Returns a list of Values in the order they should be processed for backward pass (gradient computation).
        This is essentially a topological sort of the computation graph.
        """
        ordered = []
        visited = set()

        def dfs(node: Value):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                ordered.append(node)

        dfs(self)
        return list(reversed(ordered))
    
    def sum(self) -> "Value":
        return Value(function=Sum(), parents=[self], needs_grad=self.needs_grad)
    
    def sin(self) -> "Value":
        return Value(function=fn.trig.Sin(), parents=[self], needs_grad=self.needs_grad)
    
    def cos(self) -> "Value":
        return Value(function=fn.trig.Cos(), parents=[self], needs_grad=self.needs_grad)

    def tan(self) -> "Value":
        return Value(function=fn.trig.Tan(), parents=[self], needs_grad=self.needs_grad)

    def sinh(self) -> "Value":
        return Value(function=fn.trig.Sinh(), parents=[self], needs_grad=self.needs_grad)

    def cosh(self) -> "Value":
        return Value(function=fn.trig.Cosh(), parents=[self], needs_grad=self.needs_grad)

    def tanh(self) -> "Value":
        return Value(function=fn.trig.Tanh(), parents=[self], needs_grad=self.needs_grad)
    
    def exp(self) -> "Value":
        return Value(function=fn.math.Exp(), parents=[self], needs_grad=self.needs_grad)
    
    def __add__(self, other: ValueOrLiteral) -> "Value":
        other = maybe_literal(other)
        return Value(function=Add(), parents=[self, other], needs_grad=self.needs_grad or other.needs_grad)
    
    def __sub__(self, other: ValueOrLiteral) -> "Value":
        return self + -other

    def __neg__(self) -> "Value":
        return self * -1
    
    def __mul__(self, other: ValueOrLiteral) -> "Value":
        """
        This is element-wise multiplication.
        """
        other = maybe_literal(other)
        return Value(function=ElemMul(), parents=[self, other], needs_grad=self.needs_grad or other.needs_grad)
    
    def __truediv__(self, other: ValueOrLiteral) -> "Value":
        other = maybe_literal(other)
        return Value(function=Div(), parents=[self, other], needs_grad=self.needs_grad or other.needs_grad)
    
    def __pow__(self, other: ValueOrLiteral) -> "Value":
        other = maybe_literal(other)
        return Value(function=Pow(), parents=[self, other], needs_grad=self.needs_grad or other.needs_grad)
    
    def __matmul__(self, other: ValueOrLiteral) -> "Value":
        other = maybe_literal(other)
        return Value(function=MatMul(), parents=[self, other], needs_grad=self.needs_grad or other.needs_grad)
    
    def __abs__(self) -> "Value":
        return Value(function=fn.math.Abs(), parents=[self], needs_grad=self.needs_grad)
    
    def abs(self) -> "Value":
        return self.__abs__()
    
    def __radd__(self, other: ValueOrLiteral) -> "Value":
        return self + other
    
    def __rmul__(self, other: ValueOrLiteral) -> "Value":
        return self * other
    
    def __rtruediv__(self, other: ValueOrLiteral) -> "Value":
        other = maybe_literal(other)
        return Value(function=Div(), parents=[other, self], needs_grad=self.needs_grad or other.needs_grad)
    
    def __rpow__(self, other: ValueOrLiteral) -> "Value":
        return other ** self
    
    def __rmatmul__(self, other: ValueOrLiteral) -> "Value":
        return other @ self
    
    def __eq__(self, other: ValueOrLiteral) -> "Value":
        return np.all(np.equal(self.data, maybe_value(other)))
    
    def __req__(self, other: ValueOrLiteral) -> "Value":
        return self == other
    
    def __lt__(self, other: ValueOrLiteral) -> bool:
        return np.all(np.less(self.data, maybe_value(other)))

    def __gt__(self, other: ValueOrLiteral) -> bool:
        return np.all(np.greater(self.data, maybe_value(other)))
    
    def __le__(self, other: ValueOrLiteral) -> bool:
        return np.all(np.less_equal(self.data, maybe_value(other)))

    def __ge__(self, other: ValueOrLiteral) -> bool:
        return np.all(np.greater_equal(self.data, maybe_value(other)))
    
    def __hash__(self):
        return hash((tuple(self.data.flatten()), self.function, self.needs_grad, self.label))

def maybe_literal(x: Literal | Value) -> Value:
    """
    If necessary, converts a literal (float, int, list, NP array) into a Value.
    Otherwise, returns the value as-is.
    """
    if isinstance(x, Value):
        return x
    else:
        return Value(np.array(x, dtype=np.float64))

def maybe_value(x: Literal | Value) -> np.ndarray:
    """
    If necessary, converts a Value into a literal (float, int, list, NP array).
    Otherwise, returns the number as-is.
    """
    if isinstance(x, Value):
        return np.array(x.data, dtype=np.float64)
    else:
        return np.array(x, dtype=np.float64)