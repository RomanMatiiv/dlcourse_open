import numpy as np


def numeric_grad(f, x, index, delta=1e-5):
    """
    two point formula for numerical gradient

    my function

    f: function that receives x and computes value and gradient
    x: np array, initial point where gradient is checked


    f(x+delta) - f(x-delta)
  --------------------------
            2*delta
    """
    x_one_val = x[index]+delta
    x_two_val = x[index]-delta

    x_one = x.copy()
    x_two = x.copy()

    x_one[index] = x_one_val
    x_two[index] = x_two_val

    one_point = f(x_one)[0]
    two_point = f(x_two)[0]

    df_dx = ((one_point-two_point)/(2*delta))

    numerical_grad = x.copy()
    numerical_grad[index] = df_dx

    return numerical_grad


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''

    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)

    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]

        # TODO compute value of numeric gradient of f to idx
        grad = numeric_grad(f, x, ix, delta)
        numeric_grad_at_ix = grad[ix]

        # print(analytic_grad_at_ix)
        # print(numeric_grad_at_ix)
        # print()

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

