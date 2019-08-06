import autodiff as ad
import numpy as np
import backend as T
from tensors.synthetic_tensors import init_rand_3d

BACKEND_TYPES = ['numpy', 'ctf']

def CPD_gradient_descent(size, rank, learning_rate):

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)

        A = ad.Variable(name='A')
        B = ad.Variable(name='B')
        C = ad.Variable(name='C')

        contract_A_B = ad.einsum("ia,ja->ija", A, B)
        output_tensor = ad.einsum("ija,ka->ijk", contract_A_B, C)
        norm_error = ad.norm(output_tensor - input_tensor_val)
        loss = norm_error * norm_error

        grad_A, grad_B, grad_C = ad.gradients(loss, [A, B, C])
        executor = ad.Executor([loss, grad_A, grad_B, grad_C])

        for i in range(1000):
            loss_val, grad_A_val, grad_B_val, grad_C_val = executor.run(
                feed_dict={A: A_val, B: B_val, C: C_val})
            A_val -= learning_rate * grad_A_val
            B_val -= learning_rate * grad_B_val
            C_val -= learning_rate * grad_C_val
            print('At iteration ' + str(i) + ', the loss is: ' + str(loss_val))


if __name__ == "__main__":
    CPD_gradient_descent(size=20, rank=5, learning_rate=1e-3)
