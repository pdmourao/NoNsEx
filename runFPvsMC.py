from MCclasses import HopfieldMC as hop
import numpy as np
import FPfuncs as fp
from FPfields import NoNsEx

pert_matrix = np.random.uniform(-1, 1, size=(3, 3))

# np.random.seed(0)

non_rand_pert = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]])

neurons = 5000
K = 5
beta = 1/0.74
lmb = 0.1
rho = 0.05
H = 0
M = 1000
max_it_S = 100
max_it_P = 100
max_it_FP = 1000
error = 0
ibound = 0
av_counter = 5
quality = [0.9, 0.9, 0.9]

parallel = False

print('Defining system...')
system = hop(N=neurons, pat=K, L=3, rho=rho, M=M, sigma=quality, lmb=lmb)

initial_m, initial_n = system.mattis(system.sigma), system.ex_mags(system.sigma)

print('Initial state m')
print(initial_m)

print('Initial state n')
print(initial_n)

run_main = True
if run_main:
    print('Iterating system 1 (parallel)...')
    mag_P = system.simulate(max_it=max_it_P, error=error, beta=beta, H=H, av_counter=av_counter, parallel=True,
                            disable=False)
    print('Iterating system 1 (sequential)...')
    mag_S = system.simulate(max_it=max_it_S, error=error, beta=beta, H=H, av_counter=av_counter, parallel=False,
                            disable=False)
    print('Averages from MC')
    print('Parallel:')
    print(np.mean(mag_P[-av_counter:], axis=0))
    print('Sequential:')
    print(np.mean(mag_S[-av_counter:], axis=0))

    print_all = True
    if print_all:
        print('History')
        for idx_m, m in enumerate(mag_S):
            print(f'Iteration {idx_m}')
            print(m)
            print(f'std {np.max(np.std(mag_S[idx_m-av_counter: idx_m], axis = 0))}')
        for idx_m, m in enumerate(mag_P):
            print(f'Iteration {idx_m}')
            print(m)
            print(f'std {np.max(np.std(mag_P[idx_m - av_counter: idx_m], axis=0))}')

eps = 0
pert_matrix_seeded = np.random.uniform(-1, 1, size=(3, 3))

unpert2 = np.full(shape=(3, 3), fill_value=1 / 2)
pert = eps * non_rand_pert
initial_state = unpert2 + np.full(shape=(3, 3), fill_value=pert)
q_initial = 1

run_FP = True
if run_FP:
    print('Running fixed point')
    mFP, q, n = fp.solve(NoNsEx, initial_m, np.array([q_initial, q_initial, q_initial]), initial_n + pert, max_it=1000,
                         beta=beta, lmb=lmb, rho=rho, alpha=0, H=0, ibound=ibound, error=1e-16)

    print('Fixed point')
    print(mFP[-1])
