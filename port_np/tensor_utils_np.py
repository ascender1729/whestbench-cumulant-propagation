"""NumPy port of mlp_kprop.tensor_utils (symmetric-tensor helpers).
flop_name is a no-op here; flopscope counts the underlying numpy ops in the
real estimator. Uses numpy.einsum (optimize=True) instead of opt_einsum to
avoid a grader dependency.
"""
import numpy as np
from collections import defaultdict
from contextlib import contextmanager

from port_np._backend import (
    wrapped_add,
    wrapped_allclose,
    wrapped_copy,
    wrapped_divide,
    wrapped_einsum,
)

@contextmanager
def flop_name(*a, **k):
    yield

def is_symmetric(A, vec=None):
    A = np.asarray(A)
    if A.ndim == 0: return True
    d = A.ndim; n = A.shape[0]
    if A.shape != (n,)*d: return False
    if d <= 1: return True
    if vec is None: vec = (1,)*d
    elif len(vec) != d: raise ValueError("vec length")
    classes = defaultdict(list)
    for i,lbl in enumerate(vec): classes[lbl].append(i)
    for axes in classes.values():
        if len(axes) <= 1: continue
        axes = sorted(axes)
        if not wrapped_allclose(A, np.swapaxes(A, axes[0], axes[1])): return False
        if len(axes) == 2: continue
        perm = list(range(d))
        for j in range(len(axes)): perm[axes[j]] = axes[(j+1)%len(axes)]
        if not wrapped_allclose(A, np.transpose(A, perm)): return False
    return True

def symmetrize(A, vec=None):
    A = np.asarray(A); d = A.ndim
    if d <= 1: return A
    if len(set(A.shape)) != 1: raise ValueError("shape (n,...,n)")
    if vec is None: vec = (1,)*d
    if len(vec) != d: raise ValueError("vec length")
    classes = defaultdict(list)
    for i,lbl in enumerate(vec): classes[lbl].append(i)
    U = A
    for axes in classes.values():
        k = len(axes)
        if k <= 1: continue
        axes = sorted(axes)
        V = U
        for t in range(1, k):
            acc = wrapped_copy(V)
            for j in range(t): acc = wrapped_add(acc, np.swapaxes(V, axes[j], axes[t]))
            V = wrapped_divide(acc, t+1)
        U = V
    return U

def cached_einsum(*tensors_and_expr):
    tensors = tensors_and_expr[:-1]; expr = tensors_and_expr[-1]
    # convert 'a b, b c -> a c' (einops style, space-separated) to numpy 'ab,bc->ac'
    np_expr = expr.replace(" ", "")
    return wrapped_einsum(np_expr, *tensors)

def contract_W_basic(A, W):
    A = np.asarray(A)
    if A.ndim == 0: return A
    s = A.ndim
    A_expr = ''.join(chr(ord('a')+i) for i in range(s))
    W_expr = ','.join(chr(ord('A')+i)+chr(ord('a')+i) for i in range(s))
    out_expr = ''.join(chr(ord('A')+i) for i in range(s))
    np_expr = f'{A_expr},{W_expr}->{out_expr}'
    ret = wrapped_einsum(np_expr, A, *([W]*s))
    if is_symmetric(A): ret = symmetrize(ret)
    return ret

def expand(X, positions, d):
    assert len(positions) == X.ndim
    pos_sorted = sorted(positions); p = 0
    for ax in range(d):
        if p < len(pos_sorted) and pos_sorted[p] == ax: p += 1
        else: X = np.expand_dims(X, ax)
    return X

def is_scalar(x):
    return isinstance(x,(int,float)) or (isinstance(x,np.ndarray) and x.ndim==0)

if __name__ == "__main__":
    import torch
    from mlp_kprop.tensor_utils import symmetrize as t_sym, contract_W_basic as t_cwb, is_symmetric as t_issym
    rng = np.random.default_rng(1); maxerr = 0.0
    for _ in range(10):
        n = 8
        for d in [2,3]:
            A = rng.standard_normal((n,)*d)
            e1 = float(np.max(np.abs(symmetrize(A) - t_sym(torch.tensor(A)).numpy()))); maxerr=max(maxerr,e1)
            W = rng.standard_normal((n,n))
            e2 = float(np.max(np.abs(contract_W_basic(A,W) - t_cwb(torch.tensor(A),torch.tensor(W)).numpy()))); maxerr=max(maxerr,e2)
            As = symmetrize(A)
            assert is_symmetric(As)==bool(t_issym(torch.tensor(As))), "is_symmetric mismatch"
    print(f"tensor_utils port max abs error vs torch: {maxerr:.3e}")
    print("PASS" if maxerr<1e-10 else "FAIL")
