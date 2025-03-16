from MCclasses import HopfieldMC as hop
import numpy as np
import FPfuncs as fp
from FPfields import NoNsEx, m_in, initial_q

pert_matrix = np.random.uniform(-1, 1, size=(3, 3))

# np.random.seed(0)

non_rand_pert = np.array([[1, 0, 0],
                          [0, 0, 0],
                          [0, 0,-1]])

neurons = 5000
K = 5
beta = 1/0.36
lmb = 0.2
rho = 0.01
H = 0
M = 100
max_it_P = 20
max_it_S = max_it_P
max_it_FP = 100
error = 1e-16
ibound = 1e-20
av_counter = 5
quality = [0.9, 0.9, 0.9]



run_main = True
if run_main:
    print('Defining system...')
    system = hop(neurons=neurons, K=K, L=3, rho=rho, M=M, lmb=lmb)

    R = system.r ** 2 + (1 - system.r ** 2) / system.M
    print(R)
    print(np.linalg.norm(system.ex_av, axis = 2) / (np.sqrt(system.N*R)))



    initial_m, initial_n = system.mattis(system.sigma), system.ex_mags(system.sigma)

    print('Initial state m')
    print(initial_m)

    print('Initial state n')
    print(initial_n*(1+rho))

    compute_stuff = True
    if compute_stuff:
        print('Iterating system 1 (parallel)...')
        mag_P, exmag_P = system.simulate(max_it=max_it_P, error=error, beta=beta, H=H, av_counter=av_counter, parallel=True,
                                disable=False)
        print('Iterating system 1 (sequential)...')
        mag_S, exmag_S = system.simulate(max_it=max_it_S, error=error, beta=beta, H=H, av_counter=av_counter, parallel=False, disable=False)
        print('Averages from MC')
        print('Parallel:')
        print(np.mean(mag_P[-av_counter:], axis=0))
        print('Sequential:')
        print(np.mean(mag_S[-av_counter:], axis=0))

    print_all = True
    if print_all:

        print('History')
        for idx_m, m in enumerate(mag_S):
            print(f'Iteration {idx_m} S')
            print(m)
            print('std')
            print(np.max(np.std(mag_S[idx_m - av_counter:idx_m], axis = 0)))
            print('n')
            print(exmag_S[idx_m])
        for idx_m, m in enumerate(mag_P):
            print(f'Iteration {idx_m} P')
            print('m')
            print(m)
            print('std')
            print(np.max(np.std(mag_S[idx_m - av_counter:idx_m], axis = 0)))
            print('n')
            print(exmag_P[idx_m])


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
