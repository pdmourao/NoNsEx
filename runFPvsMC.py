from MCclasses import HopfieldMC as hop, HopfieldMC_rho as hop_rho
import numpy as np
import FPfuncs as fp
from FPfields import NoNsEx

pert_matrix = np.random.uniform(-1, 1, size = (3, 3))

np.random.seed(0)

def gJprod(g_in, J_in):
    return np.transpose(np.transpose(J_in, [1, 3, 0, 2]) * g_in, [2, 0, 3, 1])

non_rand_pert = np.array([[ 1,  1,  1],
						  [ 0,  0,  0],
						  [-1, -1, -1]])

neurons = 5000
K = 5
beta = 1e10
lmb = 0
rho = 0.2
H = 0
M = 1000
max_it = 1
error = 0
ibound = 0
av_counter = 1

parallel = False

if parallel:
    dl = 'P'
else:
    dl = 'S'

print('Defining system...')
system = hop(N=neurons, pat=K, L=3, quality=(rho, M))
system_rho = hop_rho(N = neurons, pat = system.pat, blur = system.blur[0], L = 3, quality=(rho, M))

initial_m, initial_n = np.transpose(system.mattis(system.sigma)), np.transpose(system.ex_mags(system.sigma))

# print('Initial state m from 9d')
# print(initial_m)
print('Initial state m from 3d')
print(system_rho.mattis(system_rho.sigma))
# print('Initial state n from 9d')
# print(initial_n)
print('Initial state n from 3d')
print(system_rho.ex_mags(system_rho.sigma))

g = np.array([[    1, - lmb, - lmb],
              [- lmb,     1, - lmb],
              [- lmb, - lmb,     1]])
J = gJprod(g, system.J)
run_main = False
if run_main:
	print('Iterating system 1 (parallel)...')
	mag_P = system.simulate(max_it=max_it, error = error, J=J, beta = beta, H=H, av_counter = av_counter, parallel=True, disable = False)
	print('Iterating system 1 (sequential)...')
	mag_S = system.simulate(max_it=max_it, error = error, J=J, beta = beta, H=H, av_counter = av_counter, parallel=False, disable = False)
	print('Averages from MC')
	print('Parallel:')
	print(np.transpose(np.mean(mag_P[-av_counter:], axis = 0)))
	print('Sequential:')
	print(np.transpose(np.mean(mag_S[-av_counter:], axis = 0)))

print('Iterating system 1 (parallel)...')
sigma_P_rho = system_rho.simulate(max_it=max_it, error = error, beta = beta, parallel=True, disable = True)
print('Iterating system 1 (sequential)...')
sigma_S_rho = system_rho.simulate(max_it=max_it, error = error, beta = beta, parallel=False, disable = True)

mag_P_rho = np.array([system_rho.mattis(sigma) for sigma in sigma_P_rho])
mag_S_rho = np.array([system_rho.mattis(sigma) for sigma in sigma_S_rho])
n_P_rho = np.array([system_rho.ex_mags(sigma) for sigma in sigma_P_rho])
n_S_rho = np.array([system_rho.ex_mags(sigma) for sigma in sigma_S_rho])

eps = 0
pert_matrix_seeded = np.random.uniform(-1, 1, size = (3, 3))
# unpert1 = np.transpose(mattisP[0])
unpert2 = np.full(shape = (3, 3), fill_value = 1/2)
pert = eps * non_rand_pert
initial_state = unpert2 + np.full(shape = (3, 3), fill_value=pert)
q_initial = 1

print('Averages from MC, rho only')
print('Parallel:')
print(np.mean(mag_P_rho[-av_counter:], axis = 0))
print('Sequential:')
print(np.mean(mag_S_rho[-av_counter:], axis = 0))



print_all = False
for it in range(max_it+1):
	if print_all:
		print(f'\nIteration {it}...')
		print('Parallel m:')
		print(mag_P_rho[it])
		print('Parallel n:')
		print(n_P_rho[it])
		# print('Sequential m:')
		# print(mag_S_rho[it])
		# print('n')
		# print(n[it])

		# print('q')
		# print(q[it])

run_FP = False
if run_FP:
	print('Running fixed point')
	mFP, q, n = fp.solve(NoNsEx, initial_m, np.array([q_initial, q_initial, q_initial]), initial_n + pert, max_it = 1000, beta = beta, lmb = lmb, rho = rho, alpha = 0, H = 0, ibound = ibound, error = 1e-16)

	print('Fixed point')
	print(mFP[-1])
