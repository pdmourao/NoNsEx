import numpy as np
from MCclasses import HopfieldMC as hop
from tqdm import tqdm
from time import time
import json
import os
from storage import npz_file_finder
from npy_append_array import NpyAppendArray


# freqs function
# Runs the MC simulation repeatedly for the same inputs
# If it disentangles with each magnetization above a given threshold, it considers that run successful.
# Outputs the fraction of successful runs
# The point is to run its vectorized version (see below)
# for varying lambdas and temperatures

# INPUTS:
# lmb is lambda
# beta is 1/T
# systems is either a list of HopfieldMC objects or an integer
# If it is an integer, it initializes that number of HopfieldMC objects
# The cutoff is the necessary magnetization for an experiment to be considered successful
# max_it, error and H are inputs to the simulate method (See above)
# Optional beta_min, beta_max, lmb_min and lmb_max are to be used to avoid running the simulation at certain values
# For example, high temperatures take a long time, and we know they give 0
# pbar is for the progress bar






# gJprod inserts a g matrix into an already computed J
# (see in HopfieldMC class why these are separated)

def gJprod(g, J):
	return np.transpose(np.transpose(J, [1, 3, 0, 2]) * g, [2, 0, 3, 1])

def MCHop_InAndOut(neurons, K, rho, M, lmb, sigma_type, quality, noise_dif, beta, H, max_it, error, av_counter,
				   dynamic, L = 3, h = None, rngSS = np.random.SeedSequence(), prints = True, cut = False):
	t = time()
	system = hop(neurons= neurons, L = L, K= K, rho = rho, M = M, lmb = lmb, sigma_type = sigma_type, quality= quality,
				 noise_dif = noise_dif, h = h, rngSS = rngSS)
	t = time()- t
	if prints:
		print(f'System generated in {round(t, 2)} secs.')

	sim_rngSS = rngSS.spawn(1)[0]

	return system.simulate(beta = beta, H = H, max_it = max_it, error = error, av_counter = av_counter,
						   dynamic = dynamic, cut = cut, disable = prints, sim_rngSS = sim_rngSS)

def MC2d(directory, save_n, n_samples, y_values, y_arg, x_values, x_arg, dynamic, noise_dif, sigma_type, disable = False, **kwargs):

	directory = directory
	len_y = len(y_values)
	len_x = len(x_values)

	json_dict = {'dynamic': dynamic,
				 'noise_dif': noise_dif,
				 'sigma_type': sigma_type,
				 'save_m': save_n}

	inputs = {**json_dict, **kwargs, x_arg: x_values, y_arg: y_values}

	npz_files = npz_file_finder(directory = directory, prints = False, **json_dict, **inputs)
	if len(npz_files) > 1:
		print('Warning: more than 1 experiments found for given inputs.')

	try:
		file_npz = npz_files[0]
		with open(file_npz[:-3] + 'json', mode="r", encoding="utf-8") as json_file:
			data = json.load(json_file)
			entropy_from_os = int(data['entropy'])
		print('File found. Restarting.')
		samples_present = len([file for file in os.listdir(directory) if
							   file_npz[:-4] in os.path.join(directory, file) and file[-5] == 'm.npy'])
		print(f'There are {samples_present} sample(s) present')
		if n_samples == 0:
			if samples_present > 0:
				last_sample = np.load(file_npz[:-4] + f'_sample{samples_present - 1}_m.npy')
				if len(last_sample) < len_x * len_y:
					samples_present -= 1
			if samples_present > 0:
				n_samples = samples_present
			else:
				raise Exception('Provide higher number of samples to compute.')

	except IndexError:
		print('No experiments found for given inputs. Starting one.')
		if n_samples == 0:
			raise Exception('Provide higher number of samples to compute.')
		file_npz = os.path.join(directory, f'MC2dF_{x_arg}{y_arg}{len_x * len_y}_{int(time())}.npz')
		entropy_from_os = np.random.SeedSequence().entropy
		with open(f'{file_npz[:-3]}json', mode="w", encoding="utf-8") as json_file:
			json_dict['entropy'] = str(entropy_from_os)
			json.dump(json_dict, json_file)
		np.savez(file_npz, x_arg = x_values, y_arg = y_values, **kwargs)

	mattis = np.zeros((n_samples, len_x, len_y, 3, 3))
	mattis_ex = np.zeros((n_samples, len_x, len_y, 3, 3))

	t0 = time()
	rng_seeds = np.random.SeedSequence(entropy_from_os).spawn(len_x * len_y)
	print(f'Generated seeds for simulate in {round(time() - t0, 3)} s.')

	for idx_s in range(n_samples):
		t = time()
		print(f'\nSolving system {idx_s + 1}/{n_samples}...')

		file_npy_m = file_npz[:-4] + f'_sample{idx_s}_m.npy'
		file_npy_n = file_npz[:-4] + f'_sample{idx_s}_n.npy'

		try:
			mattis_flat = np.load(file_npy_m)
			if save_n:
				mattis_flat_ex = np.load(file_npy_n)
				assert len(mattis_flat_ex) == len(mattis_flat), 'Sample files corrupted. Fix or delete.'
			else:
				mattis_flat_ex = []

		except FileNotFoundError:
			mattis_flat = []
			mattis_flat_ex = []


		if len(mattis_flat) < len_x * len_y:
			if len(mattis_flat) == 0:
				print('Sample not present.')
			else:
				print(f'Sample incomplete ({len(mattis_flat)}/{len_x * len_y})')


		with tqdm(total=len_x * len_y, disable=disable) as pbar:
			for idx_x, x_v in enumerate(x_values):
				inputs[x_arg] = x_v
				for idx_y, y_v in enumerate(y_values):

					try:
						mattis[idx_s, idx_x, idx_y] = mattis_flat[idx_x * len_y + idx_y]
						if save_n:
							mattis_ex[idx_s, idx_x, idx_y] = mattis_flat_ex[idx_x * len_y + idx_y]
					except IndexError:
						inputs[y_arg] = y_v

						output_m, output_n = MCHop_InAndOut(cut = True, rngSS = rng_seeds[idx_x * len_y + idx_y],
															**inputs)

						output_m_mean = np.mean(output_m, axis=0)
						output_n_mean = np.mean(output_n, axis=0)

						mattis[idx_s, idx_x, idx_y] = output_m_mean
						mattis_ex[idx_s, idx_x, idx_y] = output_n_mean

						with NpyAppendArray(file_npy_m) as npyf:
							npyf.append(output_m_mean.reshape((1, 3, 3)))

						if save_n:
							with NpyAppendArray(file_npy_n) as npyf:
								npyf.append(output_n_mean.reshape((1, 3, 3)))

					if disable:
						print(f'{x_arg} = {round(x_v, 2)}, {y_arg} = {round(y_v, 2)} done.')
						# print(f'MaxSD = {np.max(np.std(output, axis=0))}')
						# print(f'MaxDif = {np.max(np.sum(np.diff(output, axis=0), axis=0))}')
					else:
						pbar.update(1)

		t = time() - t
		print(f'System ran in {round(t / 60)} minutes.')

	return mattis, mattis_ex


def MC2d_Lb(directory, save_n, n_samples, neurons, K, rho, M, lmb, dynamic, noise_dif, sigma_type, quality, disable = False,
			**sim_scalar_kwargs):

	directory = directory

	json_dict = {'dynamic': dynamic,
				 'noise_dif': noise_dif,
				 'sigma_type': sigma_type,
				 'save_n': save_n}

	npz_dict = {'neurons': neurons,
				'K': K,
				'rho': rho,
				'M': M,
				'lmb': lmb,
				'quality': quality}

	# Identify the y_axis array
	y_arg = None
	for item, value in sim_scalar_kwargs.items():
		if not np.isscalar(value):
			assert y_arg is None, 'Too many non-scalar arguments given to MC2d_Lb'
			y_arg = item
			y_values = value

	# Get length of input arrays
	# This will fail if no other array besides lmb is given
	len_l = len(lmb)
	len_y = len(y_values)

	npz_files = npz_file_finder(directory = directory, prints = False, **json_dict, **npz_dict, **sim_scalar_kwargs)

	if len(npz_files) > 1:
		print('Warning: more than 1 experiments found for given inputs.')

	try:
		file_npz = npz_files[0]
		with open(file_npz[:-3] + 'json', mode="r", encoding="utf-8") as json_file:
			data = json.load(json_file)
			entropy_from_os = int(data['entropy'])
		print('File found. Restarting.')
		samples_present = len([file for file in os.listdir(directory) if file_npz[:-4] in os.path.join(directory, file) and file[-5] == 'm.npy'])
		print(f'There are {samples_present} sample(s) present')
		if n_samples == 0:
			if samples_present > 0:
				last_sample = np.load(file_npz[:-4] + f'_sample{samples_present - 1}_m.npy')
				if len(last_sample) < len_l * len_y:
					samples_present -= 1
			if samples_present > 0:
				n_samples = samples_present
			else:
				raise Exception('Provide higher number of samples to compute.')

	except IndexError:
		print('No experiments found for given inputs. Starting one.')
		if n_samples == 0:
			raise Exception('Provide higher number of samples to compute.')
		file_npz = os.path.join(directory, f'MC2d_lmb{y_arg}{len_l*len_y}_{int(time())}.npz')
		entropy_from_os = np.random.SeedSequence().entropy
		with open(f'{file_npz[:-3]}json', mode="w", encoding="utf-8") as json_file:
			json_dict['entropy'] = str(entropy_from_os)
			json.dump(json_dict, json_file)
		np.savez(file_npz, **npz_dict, **sim_scalar_kwargs)

	mattis = np.zeros((n_samples, len_l, len_y, 3, 3))
	mattis_ex = np.zeros((n_samples, len_l, len_y, 3, 3))

	for idx_s in range(n_samples):
		t = time()
		print(f'\nSolving system {idx_s + 1}/{n_samples}...')

		file_npy_m = file_npz[:-4] + f'_sample{idx_s}_m.npy'
		file_npy_n = file_npz[:-4] + f'_sample{idx_s}_n.npy'

		try:
			mattis_flat = np.load(file_npy_m)
			if save_n:
				mattis_flat_ex = np.load(file_npy_n)
				assert len(mattis_flat_ex) == len(mattis_flat), 'Sample files corrupted. Fix or delete.'
			else:
				mattis_flat_ex = []

		except FileNotFoundError:
			mattis_flat = []
			mattis_flat_ex = []

		entropy = (entropy_from_os, idx_s)

		if len(mattis_flat) < len_l*len_y:
			if len(mattis_flat) == 0:
				print('Sample not present.')
			else:
				print(f'Sample incomplete ({len(mattis_flat)}/{len_l*len_y})')
			rngSS = np.random.SeedSequence(entropy)
			system = hop(neurons=neurons, K=K, L=3, rho=rho, M=M, noise_dif=noise_dif, sigma_type=sigma_type,
						 quality=quality, rngSS = rngSS)
			t0 = time()
			print(f'Initialized system in {round(t0 - t, 3)} s.')
			rng_seeds = rngSS.spawn(len_l * len_y)
			print(f'Generated seeds for simulate in {round(time() - t0, 3)} s.')

		else:
			system = None
			rng_seeds = None

		new_inputs = dict(sim_scalar_kwargs)

		with tqdm(total=len_l * len_y, disable=disable) as pbar:

			for idx_l, lmb_v in enumerate(lmb):
				if len(mattis_flat) < (idx_l+1)*len_y:
					g = np.array([[1, - lmb_v, - lmb_v],
							  [- lmb_v, 1, - lmb_v],
							  [- lmb_v, - lmb_v, 1]])
					J_lmb = gJprod(g, system.J)
				else:
					J_lmb = None

				for idx_y, y_v in enumerate(y_values):

					try:
						mattis[idx_s, idx_l, idx_y] = mattis_flat[idx_l * len_y + idx_y]
						if save_n:
							mattis_ex[idx_s, idx_l, idx_y] = mattis_flat_ex[idx_l * len_y + idx_y]
					except IndexError:
						new_inputs[y_arg] = y_v
						output_m, output_n = system.simulate(J=J_lmb, dynamic=dynamic, cut=True,
												 sim_rngSS = rng_seeds[idx_l * len_y + idx_y], **new_inputs)
						output_m_mean = np.mean(output_m, axis=0)
						output_n_mean = np.mean(output_m, axis=0)

						mattis[idx_s, idx_l, idx_y] = output_m_mean
						mattis_ex[idx_s, idx_l, idx_y] = output_n_mean

						with NpyAppendArray(file_npy_m) as npyf:
							npyf.append(output_m_mean.reshape((1, 3, 3)))
						if save_n:
							with NpyAppendArray(file_npy_n) as npyf:
								npyf.append(output_n_mean.reshape((1, 3, 3)))

					if disable:
						print(f'lmb = {round(lmb_v, 2)}, {y_arg} = {round(y_v, 2)} done.')
						# print(f'MaxSD = {np.max(np.std(output, axis=0))}')
						# print(f'MaxDif = {np.max(np.sum(np.diff(output, axis=0), axis=0))}')
					else:
						pbar.update(1)

		t = time() - t
		print(f'System ran in {round(t / 60)} minutes.')

	return mattis, mattis_ex

def MC2d_Lb_old(neurons, K, rho, M, H, lmb, beta, max_it, error, parallel, noise_dif, sigma_type, quality, av_counter = 10, disable = False):

	if parallel:
		dynamic = 'parallel'
	else:
		dynamic = 'sequential'

	system = hop(neurons=neurons, K=K, L=3, rho = rho, M = M, noise_dif=noise_dif, sigma_type = sigma_type, quality= quality)

	len_l = len(lmb)
	len_b = len(beta)
	mattisses = np.zeros(shape=(len_l, len_b, 3, 3))

	with tqdm(total=len_l*len_b, disable=disable) as pbar:

		for idx_l, lmb_v in enumerate(lmb):
			g = np.array([[      1, - lmb_v, - lmb_v],
						  [- lmb_v,       1, - lmb_v],
						  [- lmb_v, - lmb_v,       1]])
			J_lmb = gJprod(g, system.J)

			for idx_b, beta_v in enumerate(beta):
				output = np.array(system.simulate(av_counter=av_counter, error=error, J=J_lmb, beta = beta_v, H=H,
												  dynamic = dynamic, max_it = max_it, cut = True)[0])

				mattisses[idx_l, idx_b] = np.mean(output, axis = 0)

				if disable:
					print(f'lmb = {round(lmb_v, 2)}, b = {round(beta_v, 2)} done.')
					print(f'MaxSD = {np.max(np.std(output, axis = 0))}')
					print(f'MaxDif = {np.max(np.sum(np.diff(output, axis = 0), axis=0))}')
				else:
					pbar.update(1)

	return mattisses

def MC1d_beta_old(neurons, K, rho, M, H, lmb, beta, max_it, error, quality, parallel, noise_dif, random_systems = True, av_counter = 10, sigma_type = 'mix', disable = False):

	mattisses = np.zeros(shape=(len(beta), 3, 3))

	if parallel:
		dynamic = 'parallel'
	else:
		dynamic = 'sequential'

	if random_systems:
		print('Generating systems...')
		systems = [hop(L=3, noise_dif=noise_dif, neurons= neurons, K= K, lmb = lmb, rho = rho, M = M, sigma_type = sigma_type, quality= quality) for _ in tqdm(beta)]
	else:
		print('Generating system...')
		systems = hop(L=3, noise_dif=noise_dif, neurons= neurons, K= K, lmb = lmb, rho = rho, M = M, sigma_type = sigma_type, quality= quality)

	for idx_b, beta_value in enumerate(tqdm(beta, disable=disable)):
		t = time()
		if random_systems:
			system = systems[idx_b]
		else:
			system = systems

		output = np.array(system.simulate(beta=beta_value, dynamic = dynamic, cut=False, H=H, max_it=max_it, error=error,
										  av_counter=av_counter)[0])

		mattisses[idx_b] = np.mean(output[-av_counter:], axis=0)

		if disable:
			print(f'\nT = {round(1/beta_value, 2)} done.')
			print(f'Output after {len(output)-1} iterations ({round(time() - t, 2)}s):')
			print(mattisses[idx_b])

	return mattisses

def pat_id(m, cutoff_rec, cutoff_mix):
	for idx, mag in enumerate(m):
		if mag > cutoff_rec:
			return idx
		if mag < -cutoff_rec:
			return -idx
	if np.all(1/2 - cutoff_mix < m) and np.all(m < 1/2 + cutoff_mix):
		return 'mix'
	if np.all(1/2 - cutoff_mix < np.abs(m)) and np.all(np.abs(m) < 1/2 + cutoff_mix):
		return 'mix_signed'
	return None

def mags_id(m, cutoff_rec, cutoff_mix):
	ids = [pat_id(line, cutoff_rec, cutoff_mix) for line in m]
	if all([ident == 'mix' for ident in ids]):
		return 'mix'
	if all([ident in ['mix', 'mix_s'] for ident in ids]):
		return 'mix_signed'
	if any([isinstance(ident, int) for ident in ids]):
		pats_recovered = [ident for ident in ids if isinstance(ident, int)]
		n_patterns = len(set(np.abs(pats_recovered)))
		signed = '_signed' if len(set(pats_recovered)) > n_patterns else ''
		inc = '_inc' if len(pats_recovered) < len(m) else ''
		return f'{n_patterns}pats{signed}{inc}'
	return 'other'

