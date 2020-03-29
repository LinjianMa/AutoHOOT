import autodiff as ad
import backend as T
from tensors.synthetic_tensors import init_rand_tucker
from examples.tucker import TuckerGraph, tucker_als, tucker_als_shared_exec

BACKEND_TYPES = ['numpy', 'ctf', 'tensorflow']


def tucker_3d_expected_vals(A_val_list, X_val, rank):
    A1_val, A2_val, A3_val = A_val_list

    # expected values
    # ttmc: tensor times matrix chain
    ttmc = T.einsum("abc,bk,cl->akl", X_val, A2_val, A3_val)
    ttmc_inner = T.einsum("akl,bkl->ab", ttmc, ttmc)
    mat, _, _ = T.svd(ttmc_inner)
    A1_val = mat[:, :rank]

    ttmc = T.einsum("abc,ak,cl->kbl", X_val, A1_val, A3_val)
    ttmc_inner = T.einsum("kbl,kcl->bc", ttmc, ttmc)
    mat, _, _ = T.svd(ttmc_inner)
    A2_val = mat[:, :rank]

    ttmc = T.einsum("abc,ak,bl->klc", X_val, A1_val, A2_val)
    ttmc_inner = T.einsum("klc,kld->cd", ttmc, ttmc)
    mat, _, _ = T.svd(ttmc_inner)
    A3_val = mat[:, :rank]

    core_val = T.einsum("abc,ak,bl,cm->klm", X_val, A1_val, A2_val, A3_val)

    return [A1_val, A2_val, A3_val], core_val


def tucker_4d_expected_vals(A_val_list, X_val, rank):
    A1_val, A2_val, A3_val, A4_val = A_val_list

    # expected values
    # ttmc: tensor times matrix chain
    ttmc = T.einsum("abcd,bk,cl,dm->aklm", X_val, A2_val, A3_val, A4_val)
    ttmc_inner = T.einsum("aklm,bklm->ab", ttmc, ttmc)
    mat, _, _ = T.svd(ttmc_inner)
    A1_val = mat[:, :rank]

    ttmc = T.einsum("abcd,ak,cl,dm->kblm", X_val, A1_val, A3_val, A4_val)
    ttmc_inner = T.einsum("kblm,kclm->bc", ttmc, ttmc)
    mat, _, _ = T.svd(ttmc_inner)
    A2_val = mat[:, :rank]

    ttmc = T.einsum("abcd,ak,bl,dm->klcm", X_val, A1_val, A2_val, A4_val)
    ttmc_inner = T.einsum("klcm,kldm->cd", ttmc, ttmc)
    mat, _, _ = T.svd(ttmc_inner)
    A3_val = mat[:, :rank]

    ttmc = T.einsum("abcd,ak,bl,cm->klmd", X_val, A1_val, A2_val, A3_val)
    ttmc_inner = T.einsum("klmd,klme->de", ttmc, ttmc)
    mat, _, _ = T.svd(ttmc_inner)
    A4_val = mat[:, :rank]

    core_val = T.einsum("abcd,ak,bl,cm,dn->klmn", X_val, A1_val, A2_val,
                        A3_val, A4_val)

    return [A1_val, A2_val, A3_val, A4_val], core_val


def test_tucker():
    dim, size, rank = 3, 5, 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        tg = TuckerGraph(dim, size, rank)
        executor = ad.Executor([tg.residual])

        A_val_list, core_val, X_val = init_rand_tucker(dim, size, rank)

        feed_dict = dict(zip(tg.A_list, A_val_list))
        feed_dict.update({tg.core: core_val, tg.X: X_val})

        residual_val, = executor.run(feed_dict=feed_dict)

        expect_residual_val = T.einsum('ae,bf,cg,efg->abc', *A_val_list,
                                       core_val) - X_val

        assert T.norm(residual_val - expect_residual_val) < 1e-8


def test_tucker_als_3d():
    dim, size, rank = 3, 5, 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_val = init_rand_tucker(dim, size, rank)
        A_val_list, _, X_val = input_val

        A_val_list_ad, core_val_ad, _ = tucker_als(dim, size, rank, 1,
                                                   input_val)

        A_val_list, core_val = tucker_3d_expected_vals(A_val_list, X_val, rank)
        A1_val, A2_val, A3_val = A_val_list

        assert T.norm(A_val_list_ad[0] - A1_val) < 1e-8
        assert T.norm(A_val_list_ad[1] - A2_val) < 1e-8
        assert T.norm(A_val_list_ad[2] - A3_val) < 1e-8
        assert T.norm(core_val_ad - core_val) < 1e-8


def test_tucker_als_shared_exec_3d():
    dim, size, rank = 3, 5, 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_val = init_rand_tucker(dim, size, rank)
        A_val_list, _, X_val = input_val

        A_val_list_ad, core_val_ad, _ = tucker_als_shared_exec(
            dim, size, rank, 1, input_val)

        A_val_list, core_val = tucker_3d_expected_vals(A_val_list, X_val, rank)
        A1_val, A2_val, A3_val = A_val_list

        assert T.norm(A_val_list_ad[0] - A1_val) < 1e-8
        assert T.norm(A_val_list_ad[1] - A2_val) < 1e-8
        assert T.norm(A_val_list_ad[2] - A3_val) < 1e-8
        assert T.norm(core_val_ad - core_val) < 1e-8


def test_tucker_als_4d():
    dim, size, rank = 4, 5, 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_val = init_rand_tucker(dim, size, rank)
        A_val_list, _, X_val = input_val


        A_val_list_ad, core_val_ad, _ = tucker_als(dim, size, rank, 1,
                                                   input_val)

        A_val_list, core_val = tucker_4d_expected_vals(A_val_list, X_val, rank)
        A1_val, A2_val, A3_val, A4_val = A_val_list

        assert T.norm(A_val_list_ad[0] - A1_val) < 1e-8
        assert T.norm(A_val_list_ad[1] - A2_val) < 1e-8
        assert T.norm(A_val_list_ad[2] - A3_val) < 1e-8
        assert T.norm(A_val_list_ad[3] - A4_val) < 1e-8
        assert T.norm(core_val_ad - core_val) < 1e-8


def test_tucker_als_shared_exec_4d():
    dim, size, rank = 4, 5, 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_val = init_rand_tucker(dim, size, rank)
        A_val_list, _, X_val = input_val

        A_val_list_ad, core_val_ad, _ = tucker_als_shared_exec(
            dim, size, rank, 1, input_val)

        A_val_list, core_val = tucker_4d_expected_vals(A_val_list, X_val, rank)
        A1_val, A2_val, A3_val, A4_val = A_val_list

        assert T.norm(A_val_list_ad[0] - A1_val) < 1e-8
        assert T.norm(A_val_list_ad[1] - A2_val) < 1e-8
        assert T.norm(A_val_list_ad[2] - A3_val) < 1e-8
        assert T.norm(A_val_list_ad[3] - A4_val) < 1e-8
        assert T.norm(core_val_ad - core_val) < 1e-8
