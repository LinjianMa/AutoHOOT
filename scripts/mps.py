import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize_pytorch import TNOptimizer
import opt_einsum as oe
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

L = 10
D = 20
niter = 15

psi0 = qtn.MPS_rand_state(L, bond_dim=D)

psi0.graph()

H = qtn.MPO_ham_heis(L)

VIEW_CONTRACTIONS = False  # set to "True" if you want to view contraction path info via opt_einsum


def normalize(psi):
    fac = (psi & psi).contract(all, optimize='auto')
    return psi.multiply(1.0 / fac**0.5, spread_over=L)


def energy(psi, H):
    energy = qtn.TensorNetwork(qtn.align_TN_1D(psi, H, psi))
    if VIEW_CONTRACTIONS:
        info = energy.contract(all,
                               optimize=oe.RandomGreedy(minimize='size',
                                                        max_repeats=128),
                               get='path-info')
        print(info)
        exit()
    else:
        e = energy.contract(all, optimize='auto')
        return e


tnopt = TNOptimizer(tn=psi0,
                    loss_fn=energy,
                    norm_fn=normalize,
                    loss_constants={'H': H},
                    progbar=True,
                    optimizer='LBFGS',
                    learning_rate=1.0)

gs = tnopt.optimize(niter)
