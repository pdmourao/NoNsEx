from MCclasses import HopfieldMC as hop, HopfieldMC_rho as hop_rho
import numpy as np
from MCfuncs import gJprod
import FPfuncs as fp
from FPfields import NoNsEx, HopEx

neurons = 5000
K = 5
beta = 10
lmb = 0
rho = 0.25
H = 0
M = 1000
max_it = 300
error = 0
ibound = 0

parallel = False

if parallel:
    dl = 'P'
else:
    dl = 'S'

# print('Defining system...')
# system = hop(N=neurons, pat=K, L=3, quality=(rho, M))
# system_rho = hop_rho(N=neurons, L=3, K = system.pat, beta = beta, quality=(rho, M), blur = system.blur[0])

g = np.array([[    1, - lmb, - lmb],
              [- lmb,     1, - lmb],
              [- lmb, - lmb,     1]])
# J_lmb = gJprod(g, system.J)

# is_ = system.sigma
# is_rho = system_rho.sigma

# av_counter = 50
#print('Iterating system 1 (parallel)...')
#outputP, mag_avP = system.simulate(max_it=max_it, error = error, J=J_lmb, T=1 / beta, H=H, av_counter = av_counter, parallel=True)
# print('Iterating system 1 (sequential)...')
# outputS, mag_avS = system.simulate(max_it=max_it, error = error, J=J_lmb, T=1 / beta, H=H, av_counter = av_counter, parallel=False)
# print('Iterating system 2 (parallel)...')
# outputP_rho = system_rho.simulate(max_it=max_it, error = error, parallel=True)
# print(np.shape(outputP_rho))
# print('Iterating system 2 (sequential)...')
# outputS_rho = system_rho.simulate(max_it=max_it, error = error, parallel=False)

# mattisP = np.array([system.mattis(output) for output in outputP])
# mattisS = np.array([system.mattis(output) for output in outputS])
# mattisP_rho = np.array([system_rho.mattis(output) for output in outputP_rho])
# mattisS_rho = np.array([system_rho.mattis(output) for output in outputS_rho])
# print(np.shape(mattisP_rho))

eps = 1e-10
# unpert1 = np.transpose(mattisP[0])
unpert2 = np.full(shape = (3, 3), fill_value = 1/2)
pert = np.array([eps, -eps, -eps])
initial_state = unpert2 + np.full(shape = (3, 3), fill_value=pert)
print('FP Initial state')
print(initial_state)
# print('MC initial state')
# print(unpert1)
q_initial = 1
mattisFP, q, n = fp.solve(NoNsEx, initial_state, np.array([q_initial, q_initial, q_initial]), beta = beta, lmb = lmb, rho = rho, alpha = 0, H = 0, max_it = max_it, ibound = ibound)
mattisFP_rho, q_rho, n_rho = fp.solve(HopEx, initial_state[0], q_initial, beta = beta, rho = rho, alpha = 0, H = 0, max_it = max_it, ibound = ibound)

it_conv = 0
print_all = True
for it, mP in enumerate(mattisFP):
	if print_all:
		# print(f'\nIteration {it}...')
		# print('Parallel:')
		# print(np.transpose(mattisP[it]))
		# print('Parallel (rho only):')
		# print(mattisP_rho[it])
		# print('Sequential:')
		# print(np.transpose(mattisS[it]))
		# print('Sequential (rho only):')
		# print(mattisS_rho[it])
		print('m')
		print(mattisFP[it])
		print(mattisFP_rho[it])
		print('n')
		print(n[it])
		print(n_rho[it])
		print('q')
		print(q[it])
		print(q_rho[it])
	if it>1:
		if np.linalg.norm(mattisFP[it-1] - mattisFP[it-2])<1e-32 and it_conv == 0:
			it_conv = it


# print('Fixed point:')
# print(mattisFP[-1])
# print(q[-1])
# print('Fixed point (rho only):')
# print(mattisFP_rho[-1])
# print('Averages')
# print('Parallel:')
# print(np.transpose(np.average(mattisP[-av_counter:], axis = 0)))
# print(np.transpose(mag_avP))
# print('Parallel (rho only):')
# print(np.average(mattisP_rho[-av_counter:], axis = 0))
# print('Sequential:')
# print(np.transpose(np.average(mattisS[-av_counter:], axis = 0)))
# print('Sequential (rho only):')
# print(np.average(mattisS_rho[-av_counter:], axis = 0))
# print('Variances')
# print('Parallel:')
# print(np.transpose(np.var(mattisP[ - av_counter:], axis=0)))
# print('Parallel (rho only):')
# print(np.var(mattisP_rho[ - av_counter:], axis=0))
# print('Sequential:')
# print(np.transpose(np.var(mattisS[it - av_counter:it], axis=0)))
# print('Sequential (rho only):')
# print(np.var(mattisS_rho[it - av_counter:it], axis=0))


print(f'FP converged at iteration {it_conv}')