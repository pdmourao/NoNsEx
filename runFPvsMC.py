from MCclasses import HopfieldMC as hop
from MCfuncs import MCHop_InAndOut
import numpy as np
import FPfuncs as fp
from FPfields import NoNsEx, m_in, initial_q

pert_matrix = np.random.uniform(-1, 1, size=(3, 3))

# np.random.seed(0)

non_rand_pert = np.array([[1, 0, 0],
                          [0, 0, 0],
                          [0, 0,-1]])
kwargs = {'neurons': 5000,
          'K': 50,
          'beta': np.inf,
          'lmb': 0,
          'rho': 0,
          'H': 0.7,
          'M': 1,
          'max_it': 20,
          'dynamic': 'parallel',
          'error': 1,
          'av_counter': 1,
          'sigma_type': 'dis',
          'quality': [1, 1, 1],
          'noise_dif': False}

output_m, output_n = MCHop_InAndOut(**kwargs)
print(output_m)



eps = 1e-8
pert_matrix_seeded = np.random.uniform(-1, 1, size=(3, 3))

unpert2 = np.full(shape=(3, 3), fill_value=1 / 2)
pert = eps * non_rand_pert
initial_state = unpert2 + np.full(shape=(3, 3), fill_value=pert)
q_initial = 1

run_FP = False
if run_FP:
    print('Running fixed point')
    mFP, q, n = fp.solve(NoNsEx, m_in(0.4)+eps*pert, np.array([q_initial, q_initial, q_initial]), max_it=max_it_FP,
                         beta=beta, lmb=lmb, rho=rho, alpha=0, H=0, ibound=ibound, error=error)

    print('Fixed point')
    for idx, m in enumerate(mFP):
        print(f'Iteration {idx+1}')
        print(m)
        m_av_nd = np.sum(m - np.diag(np.diag(m)))/6
        m_av_d = np.mean(np.diag(m))
        print('m_dif:')
        print(m - np.full((3, 3), fill_value = m_av_nd) + np.diag(np.full(3, fill_value = m_av_nd)) - np.diag(np.full(3, fill_value = m_av_d)))
        # print('n')
        # print(n[idx])
