import numpy as np
from capygrad import Value
from capygrad import fn

def test_ops_forward():
    # Basic sanity checks
    assert Value(1) == 1, "Sanity check for scalar equals."
    assert Value([1, 2, 3]) == [1, 2, 3], "Sanity check for vector equals."
    assert Value([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "Sanity check for matrix equals."
    
    # Simple arithmetic
    assert Value(2) + Value(2) == 4, "Addition"
    assert Value(3) - 6 == -3, "Subtraction"
    assert 5 * Value(4) == Value(20), "Multiplication"
    assert Value(2) ** 8 == 256, "Power"
    assert 2 / Value(8) == 0.25, "Division"
    assert -Value(3) == -3, "Negation"
    
    # Tensor ops
    assert Value(function=fn.tensor.Stack(), parents=[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "Stack operation"
    
    # Trig ops
    assert np.isclose(Value(0).sin(), 0), "Sine of 0"
    assert np.isclose(Value(np.pi/2).sin(), 1), "Sine of pi/2"
    
    assert np.isclose(Value(0).cos(), 1), "Cosine of 0"
    assert np.isclose(Value(np.pi).cos(), -1), "Cosine of pi"
    
    assert np.isclose(Value(0).tan(), 0), "Tangent of 0"
    assert np.isclose(Value(np.pi/4).tan(), 1), "Tangent of pi/4"
    
    assert np.isclose(Value(0).sinh(), 0), "Hyperbolic sine of 0"
    assert np.isclose(Value(1).sinh(), np.sinh(1)), "Hyperbolic sine of 1"
    
    assert np.isclose(Value(0).cosh(), 1), "Hyperbolic cosine of 0"
    assert np.isclose(Value(1).cosh(), np.cosh(1)), "Hyperbolic cosine of 1"
    
    assert np.isclose(Value(0).tanh(), 0), "Hyperbolic tangent of 0"
    assert np.isclose(Value(1).tanh(), np.tanh(1)), "Hyperbolic tangent of 1"
    
    # Exp op
    assert np.isclose(Value(0).exp(), 1), "Exponential of 0"
    assert np.isclose(Value(1).exp(), np.e), "Exponential of 1"
    
    # Absolute value
    assert Value(-5).abs() == 5, "Absolute value of negative number"
    assert Value(3.14).abs() == 3.14, "Absolute value of positive number"
    assert Value(0).abs() == 0, "Absolute value of zero"
    # Test with vectors
    assert np.all(Value([-1, 2, -3]).abs() == [1, 2, 3]), "Absolute value of array"
    # Test with matrices
    assert np.all(Value([[-1, 2], [-3, 4]]).abs() == [[1, 2], [3, 4]]), "Absolute value of nested array"

def test_ops_backward():
    # Addition
    x = Value(2, needs_grad=True)
    y = Value(3, needs_grad=True)
    z = x + y
    z.backward()
    assert x.grad == 1, "Gradient of addition (x)"
    assert y.grad == 1, "Gradient of addition (y)"

    # Multiplication
    x = Value(2, needs_grad=True)
    y = Value(3, needs_grad=True)
    z = x * y
    z.backward()
    assert x.grad == 3, "Gradient of multiplication (x)"
    assert y.grad == 2, "Gradient of multiplication (y)"

    # Power function
    x = Value(2, needs_grad=True)
    y = Value(-3)
    z = x ** y
    z.backward()
    assert x.grad == -3 * (2 ** -4), "Gradient of power function"
    assert np.isclose(y.grad, 2**-3 * np.log(2)), "Gradient of power function (y)"

    # Subtraction
    x = Value(5, needs_grad=True)
    y = Value(3, needs_grad=True)
    z = x - y
    z.backward()
    assert x.grad == 1, "Gradient of subtraction (x)"
    assert y.grad == -1, "Gradient of subtraction (y)"

    # Division
    x = Value(6, needs_grad=True)
    y = Value(2, needs_grad=True)
    z = x / y
    z.backward()
    assert x.grad == 0.5, "Gradient of division (x)"
    assert y.grad == -3/2, "Gradient of division (y)"

    # Sine
    x = Value(np.pi/4, needs_grad=True)
    z = x.sin()
    z.backward()
    assert np.isclose(x.grad, np.cos(np.pi/4)), "Gradient of sine"

    # Exponential
    x = Value(2, needs_grad=True)
    z = x.exp()
    z.backward()
    assert np.isclose(x.grad, np.exp(2)), "Gradient of exponential"

    # Absolute value
    x = Value(-3, needs_grad=True)
    z = x.abs()
    z.backward()
    assert x.grad == -1, "Gradient of absolute value (negative input)"

    x = Value(3, needs_grad=True)
    z = x.abs()
    z.backward()
    assert x.grad == 1, "Gradient of absolute value (positive input)"

    # Composite function
    x = Value(2, needs_grad=True)
    y = Value(3, needs_grad=True)
    z = (x**3 * y).sin() + x * 2
    z.backward()
    assert np.isclose(x.grad, 2 + 36 * np.cos(24)), "Gradient of composite function (x)"
    assert np.isclose(y.grad, 8 * np.cos(24)), "Gradient of composite function (y)"
