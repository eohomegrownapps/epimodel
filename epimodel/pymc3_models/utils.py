import theano.tensor as T
import numpy as np


def array_stats(d):
    """
    Compute the mean, standard deviation and (0.05th, 0.95th) quantiles of an array.

    Parameters
    ----------
    d : Array of numeric values

    Returns
    -------
    A string with the following format:
        ``<mean> std=<standard deviation> (<0.05th quantile> .. <0.95th quantile>)``
    e.g. ``3 std=1.414 (1.2 .. 4.8)``
    """
    d = np.array(d)
    return (
        f"{d.mean():.3g} std={d.std():.3f} "
        f"({np.quantile(d, 0.05):.3g} .. {np.quantile(d, 0.95):.3g})"
    )


def shift_right(t, dist, axis, pad=0.0):
    """
    Return the signal shifted right by dist along given axis, padded by `pad`.

    Parameters
    ----------
    t : Array of numeric values
    dist : int
    axis : int
    pad : numeric, optional
        Padding value (default: 0.0)

    Returns
    -------
    theano.tensor.TensorVariable
    """
    assert dist >= 0
    t = T.as_tensor(t)
    if dist == 0:
        return t
    p = T.ones_like(t) * pad

    # Slices
    ts = [slice(None)] * t.ndim
    ts[axis] = slice(None, -dist)  # only for dim > 0

    ps = [slice(None)] * t.ndim
    ps[axis] = slice(None, dist)

    res = T.concatenate((p[ps], t[ts]), axis=axis)
    return res


def convolution(t, weights, axis):
    """
    Computes a linear convolution of tensor by weights.
    
    The result is res[.., i, ..] = w[0] * res[.., i, ..]

    Parameters
    ----------
    t : Array of numeric values
    weights : Array of numeric values
    axis : int

    Returns
    -------
    theano.tensor.TensorVariable
    """
    t = T.as_tensor(t)
    res = T.zeros_like(t)
    for i, dp in enumerate(weights):
        res = res + dp * shift_right(t, dist=i, axis=axis, pad=0.0)
    return res


def geom_convolution(t, weights, axis):
    """
    Computes a linear convolution of log(tensor) by weights, returning exp(conv_res).
    
    Can be also seen as geometrical convolution.
    The result is res[.., i, ..] = w[0] * res[.., i, ..]

    Parameters
    ----------
    t : Array of numeric values
    weights : Array of numeric values
    axis : int

    Returns
    -------
    theano.tensor.TensorVariable
    """
    t = T.as_tensor(t)
    res = T.ones_like(t)
    for i, dp in enumerate(weights):
        res = res * shift_right(t, dist=i, axis=axis, pad=1.0) ** dp
    return res
