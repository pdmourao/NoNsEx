from MCclasses import HopfieldMC as hop
from MCfuncs import MCHop_InAndOut
import numpy as np
import FPfuncs as fp
from FPfields import NoNsEx, m_in, initial_q


kwargs = {'beta': 10,
          'lmb': 0.132,
          'rho': 0.294,
          'H': 0}

kwargs_MC = {**kwargs,
             'neurons': 5000,
             'K': 50,
             'M': 1,
             'mixM': 0,
             'max_it': 20,
             'dynamic': 'sequential',
             'error': 1,
             'av_counter': 1,
             'sigma_type': 'dis',
             'quality': [1, 1, 1],
             'noise_dif': False}

kwargs_FP = {**kwargs,
             'alpha': 0,
             'max_it': 10000,
             'error': 1e-16,
             'ibound': 0}


# output_m, output_n = MCHop_InAndOut(**kwargs_MC)
# [print(m) for m in output_m]


pert_matrix = np.random.uniform(-1, 1, size=(3, 3))
np.random.seed(1)

non_rand_pert = np.array([[ 1,-1,-1],
                          [-1, 1,-1],
                          [-1,-1, 1]])
pert_matrix_seeded = np.random.uniform(-1, 1, size=(3, 3))

eps = 1e-5
pert = eps * pert_matrix
m_initial = m_in(4.5/10)+pert
print(f'Initial m:')
print(m_initial)
maxed_it, output_list = fp.iterator(m_initial, initial_q, dif_length = 0, field = NoNsEx, **kwargs_FP)
print(output_list[-1][0])
