import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_core import rand_uuid
from quimb.core import make_immutable
import opt_einsum as oe
from autoray import numpy as np
import itertools
import functools
import matplotlib.pyplot as plt
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ## Some simple functions to help in 2D


def gen_2d_bond_pairs(n, m):
    r"""Generate all the bonds for a grid.
                 ...

          |       |       |
          |       |       |
        (1,0)---(1,1)---(1,2)---
          |       |       |
          |       |       |
        (1,0)---(1,1)---(1,2)---  ...
          |       |       |
          |       |       |
        (0,0)---(0,1)---(0,2)---

    """
    for i in range(n):
        for j in range(m):
            if i + 1 < n:
                yield (i, j), (i + 1, j)
            if j + 1 < m:
                yield (i, j), (i, j + 1)


def gen_2d_all_pairs(n, m):
    """Same as gen_2d_bond_pairs but now will generate all the pairs"""
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    n1 = n * i + j
                    n2 = n * k + l
                    if n1 < n2:
                        yield ((i, j), (k, l))


def gate(tn, X, i, j, inplace=False, tags=None):
    """Lazily apply a gate (given by array ``X`) to site ``where``
    on some 2D tensor network, assumed to have indices ``'k{},{}'``.
    """
    tn = tn if inplace else tn.copy()

    # get the outer index
    gix = f'k{i},{j}'

    bond = rand_uuid()
    tn.reindex_({gix: bond})

    tn |= qtn.Tensor(X, (gix, bond), tags=tags)

    return tn


gate_ = lambda *args, **kwargs: gate(*args, **kwargs, inplace=True)

# NOTE: variational ansatz means always higher energy, but for sqaure lattice
# with OBC, the energy is *lower* at finite sizes so these two things can cancel
# out to look like a artificially accurate energy.

# PEPS builder


def rand_peps(
        n,
        m,
        D,
        site_ind_id='k{},{}',  # naming convention for physical indices
        site_tag_id='I{},{}',  # naming convention for tagging each tensor
        phys_dim=2,
        dtype=float,
):
    """Create a random PEPs.
    """
    # create a single vector tensor at each site
    tensors = {(i, j): qtn.rand_tensor(shape=[phys_dim],
                                       inds=[site_ind_id.format(i, j)],
                                       tags=[site_tag_id.format(i, j)],
                                       dtype=dtype)
               for i in range(n) for j in range(m)}

    # create new bonds between all neighbouring pairs
    for ij1, ij2 in gen_2d_bond_pairs(n, m):
        qtn.new_bond(tensors[ij1], tensors[ij2], size=D)

    # combine into a TN
    peps = qtn.TensorNetwork(tensors.values())

    # inplace randomize all the entries to avoid |000000....0>
    peps.randomize_()
    peps.apply_to_arrays(qtn.tensor_gen.sensibly_scale)

    return peps

    # some standard functions for optimization


def norm(psi, optimize='auto', **opts):
    """This is automatically the frobenius norm of any tensor network.
    """
    return (psi & psi).contract(all, optimize=optimize, **opts)


def normalize(psi):
    """Need to run this before supplying to the energy function to
    stay in the normalized manifold.
    """
    return psi.multiply(1.0 / norm(psi)**0.5,
                        spread_over=len(psi.tensor_map),
                        inplace=False)


def local_corr(psi, s1, s2, where1, where2):
    """A single expectation like <SS> = < psi | SSpsi >
    """
    SSpsi = psi.copy()
    gate(SSpsi,
         constant(qu.spin_operator(s1, dtype=dtype)),
         *where1,
         inplace=True)
    gate(SSpsi,
         constant(qu.spin_operator(s2, dtype=dtype)),
         *where2,
         inplace=True)

    return (psi & SSpsi).contract(all, optimize='auto')


def heis_en(psi):
    """Heisenberg energy as simple sum of local correlators.
       H = \sum_{<ij>} (S^z_i S^z_j + 0.5 * S^+_i S^-_j + 0.5 * S^-_i S^+_j)
    """
    ens = []

    # try to reuse intermediate tensors
    with oe.shared_intermediates():

        for ij1, ij2 in gen_2d_bond_pairs(n, m):
            ens.append(local_corr(psi, '+', '-', ij1, ij2) / 2)
            ens.append(local_corr(psi, '-', '+', ij1, ij2) / 2)
            ens.append(local_corr(psi, 'Z', 'Z', ij1, ij2))

    return np.sum(ens)


def heis_lr_en(psi):
    """LR Heisenberg energy according to Coulomb potential. Same as above Hamiltonian,
        except all pairs of sites interact and their coupling is determind by 1/r."""
    ens = []

    with oe.shared_intermediates():

        for ij1, ij2 in gen_2d_all_pairs(n, m):
            ens.append(
                local_corr(psi, '+', '-', ij1, ij2) /
                (2 * np.sqrt((ij1[0] - ij2[0])**2 + (ij1[1] - ij2[1])**2)))
            ens.append(
                local_corr(psi, '-', '+', ij1, ij2) /
                (2 * np.sqrt((ij1[0] - ij2[0])**2 + (ij1[1] - ij2[1])**2)))
            ens.append(
                local_corr(psi, 'Z', 'Z', ij1, ij2) /
                (1 * np.sqrt((ij1[0] - ij2[0])**2 + (ij1[1] - ij2[1])**2)))

    return np.sum(ens)


if __name__ == "__main__":
    # Size of the system:

    n = 4
    m = 4
    print(f'system size = {n}x{m}')

    D = 3
    dtype = 'float32'
    psi = rand_peps(n, m, D, dtype=dtype)
    psi.graph()

    from quimb.tensor.optimize_pytorch import TNOptimizer, constant
    opts = {'optimizer': 'LBFGS', 'learning_rate': 1.0, 'progbar': True}

    # Check the size of the largest tensor produced:
    bigsize = norm(psi, get='path-info').largest_intermediate
    print(f'largest tensor size during exact contract: {bigsize}')

    # Now we can define our TN optimizer object:
    opt = TNOptimizer(
        psi.squeeze(),  # the tensor network to optimize
        loss_fn=heis_en,  # the function that computes the target scalar
        norm_fn=normalize,  # the function that normalizes the raw TN
        **opts)

    nsteps = 25
    print(f'number of optimizer steps: {nsteps}')
    opt.optimize(nsteps)
