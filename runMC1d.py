from MCclasses import HopfieldMC as hop
from MCclasses_tf import HopfieldMC_tf as hop_tf
from MCfuncs import MC1d_beta
from tqdm import tqdm
import numpy as np
from time import time
from storage import file_finder
import os
from npy_append_array import NpyAppendArray

samples = 0
sample_graph = 30
disable = False


kwargs = {'beta': 1/np.linspace(0.01, 1, 100, endpoint = True),
		  'neurons': 3000,
		  'K': 3,
		  'rho': 0.05,
		  'M': 10000,
		  'lmb': 0.2,
		  'H': 0,
		  'max_it': 10,
		  'error': 0,
		  'av_counter': 1,
		  'quality': 1/2}

parallel = False
use_tf = False
noise_dif = False
random_systems = False


if parallel and use_tf:
    dl = 'PDtf'
elif parallel and not use_tf:
    dl = 'PDnp'
else:
    dl = 'SD'

if noise_dif:
    noise_string = 'in'
else:
    noise_string = 'dn'

if random_systems:
	rand_string = 'rs'
else:
	rand_string = 'ds'

full_string = f'_{noise_string}{dl}{rand_string}_'

files = file_finder('MC1d', file_spec = full_string, **kwargs)

try:
    filename = files[0]
except IndexError:
    print('Creating new.')
    filename = os.path.join('MC1d', f'MC1d{full_string}beta{len(kwargs['beta'])}_{int(time())}.npz')


for sample in range(samples):

	t = time()
	print(f'\nSolving system {sample + 1}/{samples}...')
	mattisses = MC1d_beta(parallel = parallel, use_tf = use_tf, noise_dif = noise_dif, random_systems = random_systems,
						  disable = False, **kwargs)

	with NpyAppendArray(filename[:-1] + 'y', delete_if_exists=False) as file:
		file.append(mattisses.flatten())

	if len(files) == 0 and sample == 0:
		np.savez(filename, **kwargs)

	print('File appended.')

	t = time() - t
	print(f'System ran in {round(t / 60)} minutes.')


# CODE WHAT TO GRAPH