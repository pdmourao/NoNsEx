from tqdm import tqdm
import numpy as np
import os
from time import time
import random

n_systems = 50
M_values = [1000, 2000, 5000, 10000]
rho_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

directory = 'MC2d_FixedSN'

max_K = 5
max_N = 5000

t = time()

pats_file = os.path.join(directory, f'{n_systems}pats_{int(t)}')
np.savez(pats_file, *[np.random.choice([-1, 1], size=(max_K, max_N)) for _ in range(n_systems)])
print(f'Generated and saved {n_systems} patterns in {time()-t} seconds.')

with tqdm(total = len(M_values)*len(rho_values)) as pbar:
	for M in M_values:
		for rho in rho_values:
			r = np.sqrt(1/(rho*M + 1))
			blurs_file = os.path.join(directory, f'{n_systems}avblurs_{random.randint(a = 1, b = 10000)}')
			np.savez(blurs_file, *[np.mean(np.random.choice([-1, 1], p = [(1-r)/2, (1+r)/2],size=(M, max_K, max_N)), axis = 0)
								   for _ in range(n_systems)], M = M, rho = rho)
			pbar.update(1)

print(f'Program ran in {time()-t} seconds.')