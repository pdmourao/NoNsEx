import numpy as np
from MCclasses import HopfieldMC as hop
from tqdm import tqdm
from time import time, process_time
from multiprocessing import Pool
import json
import os
from storage import npz_file_finder, mathToPython
from npy_append_array import NpyAppendArray
from scipy.interpolate import make_interp_spline
from matplotlib import pyplot as plt


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


def SplittingExperiment(suf, n_samples, rho_values, neurons, K, M, max_it, error, av_counter, H = 0, mixM = 0, sigma_type ='mix',
                        quality = [1,1,1], dynamic = 'sequential', minlmb = 0, minT = 1e-3, disable = False):

    directory = 'ToSplit_Or_NotToSplit'

    len_rho = len(rho_values)

    interpolatorT = make_interp_spline(*mathToPython('maxT'+suf,'optpar'))
    interpolatorL = make_interp_spline(*mathToPython('maxL'+suf,'optpar'))

    inputs_sys = {'neurons': neurons, 'K': K, 'M': M, 'quality': quality, 'mixM': mixM, 'rho': rho_values}
    inputs_sys_notsplit = dict(inputs_sys)
    inputs_sys_notsplit['M'] = 3*M
    inputs_sim = {'max_it': max_it, 'error': error, 'av_counter': av_counter, 'H': H}
    inputs_json = {'suf': suf, 'sigma_type': sigma_type, 'dynamic': dynamic}
    all_inputs = {**inputs_sys, **inputs_sim, **inputs_json, 'minlmb': minlmb, 'minT': minT}

    npz_files = npz_file_finder(directory=directory, prints=False, **all_inputs)
    if len(npz_files) > 1:
        print('Warning: more than 1 experiments found for given inputs.')

    try:
        file_npz = npz_files[0]
        with open(file_npz[:-3] + 'json', mode="r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            entropy_from_os = int(data['entropy'])
        print('File found!')
        if n_samples > 0:
            print('Restarting...')
        samples_present = len([file for file in os.listdir(directory) if
                               file_npz[:-4] in os.path.join(directory, file) and 'm_split.npy' in file])
        print(f'There are {samples_present} sample(s) present')
        if n_samples == 0:
            if samples_present > 0:
                last_sample = np.load(file_npz[:-4] + f'_sample{samples_present - 1}_m_split.npy')
                if len(last_sample) < len_rho:
                    samples_present -= 1
            if samples_present > 0:
                n_samples = samples_present
            else:
                raise Exception('No samples present. Compute some first.')

    except IndexError:
        print('No experiments found for given inputs. Starting one.')
        if n_samples == 0:
            raise Exception('No complete samples present. Compute some first.')
        file_npz = os.path.join(directory, f'MCSplit_{len_rho}_{int(time())}.npz')
        entropy_from_os = np.random.SeedSequence().entropy
        np.savez(file_npz, **inputs_sys, **inputs_sim, minlmb = minlmb, minT = minT)
        with open(f'{file_npz[:-3]}json', mode="w", encoding="utf-8") as json_file:
            inputs_json['entropy'] = str(entropy_from_os)
            json.dump(inputs_json, json_file)

    mattis_split = np.zeros((n_samples, len_rho, 3, 3))
    mattis_ex_split = np.zeros((n_samples, len_rho, 3, 3))
    max_ints_split = np.zeros((n_samples, len_rho), dtype=int)
    mattis_notsplit = np.zeros((n_samples, len_rho, 3, 3))
    mattis_ex_notsplit = np.zeros((n_samples, len_rho, 3, 3))
    max_ints_notsplit = np.zeros((n_samples, len_rho), dtype=int)

    for idx_s in range(n_samples):

        t0 = time()

        entropy = (entropy_from_os, idx_s)

        rng_seeds = np.random.SeedSequence(entropy).spawn(len_rho*2)
        print(f'Generated seeds for simulate in {round(time() - t0, 3)} s.')

        t = time()
        print(f'\nSolving system {idx_s + 1}/{n_samples}...')

        file_npy_m_split = file_npz[:-4] + f'_sample{idx_s}_m_split.npy'
        file_npy_n_split = file_npz[:-4] + f'_sample{idx_s}_n_split.npy'
        file_npy_ints_split = file_npz[:-4] + f'_sample{idx_s}_ints_split.npy'
        file_npy_m_notsplit = file_npz[:-4] + f'_sample{idx_s}_m_notsplit.npy'
        file_npy_n_notsplit = file_npz[:-4] + f'_sample{idx_s}_n_notsplit.npy'
        file_npy_ints_notsplit = file_npz[:-4] + f'_sample{idx_s}_ints_notsplit.npy'

        try:

            mattis_flat_split = np.load(file_npy_m_split)
            mattis_flat_notsplit = np.load(file_npy_m_notsplit)

            mattis_flat_ex_split = np.load(file_npy_n_split)
            assert len(mattis_flat_ex_split) == len(mattis_flat_split), 'Sample files corrupted (exsplit). Fix or delete.'

            mattis_flat_ex_notsplit = np.load(file_npy_n_notsplit)
            assert len(mattis_flat_ex_notsplit) == len(mattis_flat_split), 'Sample files corrupted (exnotsplit). Fix or delete.'

            flat_ints_split = np.load(file_npy_ints_split)
            assert len(flat_ints_split) == len(mattis_flat_split), 'Sample files corrupted (ints split). Fix or delete.'

            flat_ints_notsplit = np.load(file_npy_ints_notsplit)
            assert len(flat_ints_notsplit) == len(mattis_flat_split), 'Sample files corrupted (ints split). Fix or delete.'

        except FileNotFoundError:
            mattis_flat_split = []
            mattis_flat_ex_split = []
            flat_ints_split = []
            mattis_flat_notsplit = []
            mattis_flat_ex_notsplit = []
            flat_ints_notsplit = []

        if len(mattis_flat_split) < len_rho:
            if len(mattis_flat_split) == 0:
                print('Sample not present.')
            else:
                print(f'Sample incomplete ({len(mattis_flat_split)}/{len_rho})')

        with tqdm(total=len_rho, disable=disable) as pbar:
            for idx_rho, rho_v in enumerate(rho_values):

                inputs_sys['rho'] = rho_v
                inputs_sys_notsplit['rho'] = rho_v/3
                try:
                    mattis_split[idx_s, idx_rho] = mattis_flat_split[idx_rho]
                    mattis_ex_split[idx_s, idx_rho] = mattis_flat_ex_split[idx_rho]
                    max_ints_split[idx_s, idx_rho] = flat_ints_split[idx_rho]

                    mattis_notsplit[idx_s, idx_rho] = mattis_flat_notsplit[idx_rho]
                    mattis_ex_notsplit[idx_s, idx_rho] = mattis_flat_ex_notsplit[idx_rho]
                    max_ints_notsplit[idx_s, idx_rho] = flat_ints_notsplit[idx_rho]
                except IndexError:

                    lmb_split = max(interpolatorL(rho_v), minlmb)
                    lmb_notsplit = max(interpolatorL(rho_v/3), minlmb)

                    beta_split = 1/interpolatorT(rho_v) if interpolatorT(rho_v) > minT else np.inf
                    beta_notsplit = 1 / interpolatorT(rho_v/3) if interpolatorT(rho_v/3) > minT else np.inf

                    noise_split = rng_seeds[2 * idx_rho]
                    noise_notsplit = rng_seeds[2 * idx_rho + 1]

                    split = hop(rngSS=noise_split, lmb = lmb_split, sigma_type = 'mix', noise_dif = True, **inputs_sys)

                    inputs_sys_notsplit['K'] = split.pat

                    jointex = np.full(shape = (split.L, inputs_sys_notsplit['M'], split.K, split.N),
                                        fill_value = np.concatenate(tuple(layer for layer in split.ex)))

                    notsplit = hop(rngSS=noise_notsplit, lmb = lmb_notsplit, sigma_type='mix', ex=jointex,
                                   noise_dif=False, **inputs_sys_notsplit)


                    assert np.array_equal(split.sigma, notsplit.sigma), 'Problem with the initial states.'

                    output_m_split, output_n_split, ints_split = split.simulate(beta = beta_split, dynamic = dynamic, sim_rngSS = noise_split.spawn(1)[0], cut = True, **inputs_sim)
                    output_m_notsplit, output_n_notsplit, ints_notsplit = notsplit.simulate(beta = beta_notsplit, dynamic = dynamic, sim_rngSS = noise_notsplit.spawn(1)[0], cut = True, **inputs_sim)

                    output_m_split_mean = np.mean(output_m_split, axis=0)
                    output_n_split_mean = np.mean(output_n_split, axis=0)
                    output_m_notsplit_mean = np.mean(output_m_notsplit, axis=0)
                    output_n_notsplit_mean = np.mean(output_n_notsplit, axis=0)

                    mattis_split[idx_s, idx_rho] = output_m_split_mean
                    mattis_ex_split[idx_s, idx_rho] = output_n_split_mean
                    max_ints_split[idx_s, idx_rho] = ints_split

                    mattis_notsplit[idx_s, idx_rho] = output_m_notsplit_mean
                    mattis_ex_notsplit[idx_s, idx_rho] = output_n_notsplit_mean
                    max_ints_notsplit[idx_s, idx_rho] = ints_notsplit

                    with NpyAppendArray(file_npy_m_split) as npyf:
                        npyf.append(output_m_split_mean.reshape((1, 3, 3)))

                    with NpyAppendArray(file_npy_n_split) as npyf:
                        npyf.append(output_n_split_mean.reshape((1, 3, 3)))

                    with NpyAppendArray(file_npy_ints_split) as npyf:
                        npyf.append(np.array([ints_split]))

                    with NpyAppendArray(file_npy_m_notsplit) as npyf:
                        npyf.append(output_m_notsplit_mean.reshape((1, 3, 3)))

                    with NpyAppendArray(file_npy_n_notsplit) as npyf:
                        npyf.append(output_n_notsplit_mean.reshape((1, 3, 3)))

                    with NpyAppendArray(file_npy_ints_notsplit) as npyf:
                        npyf.append(np.array([ints_notsplit]))

                if disable:
                    print(rf'$\rho$ = {round(rho_v, 2)}, split and not split systems ran to {max_ints_split[idx_s, idx_rho]} and {max_ints_notsplit[idx_s, idx_rho]} iteration(s), respectively.')
                # print(f'MaxSD = {np.max(np.std(output, axis=0))}')
                # print(f'MaxDif = {np.max(np.sum(np.diff(output, axis=0), axis=0))}')
                else:
                    pbar.update(1)

        t = time() - t
        print(f'System ran in {round(t / 60)} minutes.')

    return mattis_split, mattis_ex_split, max_ints_split, mattis_notsplit, mattis_ex_notsplit, max_ints_notsplit


def splitting_beta(entropy, beta_values, neurons, K, rho, lmb, M, max_it, error, av_counter, H = 0, mixM = 0, sigma_type ='mix',
                        quality = [1,1,1], dynamic = 'sequential', disable = True):

    len_beta = len(beta_values)

    inputs_sys = {'neurons': neurons, 'K': K, 'M': M, 'quality': quality, 'mixM': mixM, 'rho': rho, 'lmb': lmb}
    inputs_sys_notsplit = dict(inputs_sys)
    inputs_sys_notsplit['M'] = 3*M
    inputs_sys_notsplit['rho'] = rho/3
    inputs_sim = {'max_it': max_it, 'error': error, 'av_counter': av_counter, 'H': H}
    inputs_json = {'sigma_type': sigma_type, 'dynamic': dynamic}
    all_inputs = {**inputs_sys, **inputs_sim, **inputs_json}

    output_list = [np.zeros((len_beta, 3, 3)), # m_split
                     np.zeros((len_beta, 3, 3)), # n_split
                     np.zeros(len_beta, dtype=int), # ints_split
                     np.zeros((len_beta, 3, 3)), # m_notsplit
                     np.zeros((len_beta, 3, 3)), # n_notsplit
                     np.zeros(len_beta, dtype=int) # ints_notsplit
                     ]

    t = process_time()

    t0 = process_time()
    rng_seeds = np.random.SeedSequence(entropy).spawn(2)
    rng_seeds_split = rng_seeds[0].spawn(len_beta)
    rng_seeds_notsplit = rng_seeds[1].spawn(len_beta)

    if not disable:
        print(f'Generated seeds in {round(process_time() - t0, 3)} s.')
    t0 = process_time()

    if not disable:
        print('Generating systems...')

    split = hop(rngSS=rng_seeds[0], sigma_type='mix', noise_dif=True, **inputs_sys)
    inputs_sys_notsplit['K'] = split.pat
    jointex = np.full(shape=(split.L, inputs_sys_notsplit['M'], split.K, split.N), fill_value=np.concatenate(tuple(layer for layer in split.ex)))
    notsplit = hop(rngSS=rng_seeds[1], sigma_type='mix', ex=jointex, noise_dif=False, **inputs_sys_notsplit)

    if not disable:
        print(f'Generated systems in {round(process_time() - t0, 3)} s.')

    for beta_idx, beta in enumerate(tqdm(beta_values, disable = disable)):
        output_list[0][beta_idx], output_list[1][beta_idx], output_list[2][beta_idx] = split.simulate(dynamic=dynamic, beta=beta,sim_rngSS=rng_seeds_split[beta_idx], cut=True, av=True, **inputs_sim)
        output_list[3][beta_idx], output_list[4][beta_idx], output_list[5][beta_idx] = notsplit.simulate(dynamic=dynamic, beta=beta, sim_rngSS=rng_seeds_notsplit[beta_idx], cut=True, av=True, *inputs_sim)
    if not disable:
        print(f'Sample ran in {round(process_time() - t / 60)} minutes.')

    return tuple(output_list)



def SplittingExperiment_beta(n_samples, beta_values, neurons, K, rho, lmb, M, max_it, error, av_counter, H = 0, mixM = 0, sigma_type ='mix',
                        quality = [1,1,1], dynamic = 'sequential', disable = False, parallel_CPUs = False):

    directory = 'ToSplit_Or_NotToSplit_beta'

    len_beta = len(beta_values)

    inputs_sys = {'neurons': neurons, 'K': K, 'M': M, 'quality': quality, 'mixM': mixM, 'rho': rho, 'lmb': lmb}
    inputs_sys_notsplit = dict(inputs_sys)
    inputs_sys_notsplit['M'] = 3*M
    inputs_sys_notsplit['rho'] = rho/3
    inputs_sim = {'max_it': max_it, 'error': error, 'av_counter': av_counter, 'H': H}
    inputs_json = {'sigma_type': sigma_type, 'dynamic': dynamic}
    all_inputs = {**inputs_sys, **inputs_sim, **inputs_json}

    npz_files = npz_file_finder(directory=directory, prints=False, beta = beta_values, **all_inputs)
    if len(npz_files) > 1:
        print('Warning: more than 1 experiments found for given inputs.')

    try:
        file_npz = npz_files[0]
        with open(file_npz[:-3] + 'json', mode="r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            entropy_from_os = int(data['entropy'])
        print('File found!')
        if n_samples > 0:
            print('Restarting...')
        samples_present = len([file for file in os.listdir(directory) if
                               file_npz[:-4] in os.path.join(directory, file) and 'm_split.npy' in file])
        print(f'There are {samples_present} sample(s) present')
        if n_samples == 0:
            if samples_present > 0:
                n_samples = samples_present
            else:
                raise Exception('No samples present. Compute some first.')

    except IndexError:
        print('No experiments found for given inputs. Starting one.')
        if n_samples == 0:
            raise Exception('No complete samples present. Compute some first.')
        file_npz = os.path.join(directory, f'MCSplit_beta{len_beta}_{int(time())}.npz')
        entropy_from_os = np.random.SeedSequence().entropy
        np.savez(file_npz, beta = beta_values, **inputs_sys, **inputs_sim)
        with open(f'{file_npz[:-3]}json', mode="w", encoding="utf-8") as json_file:
            inputs_json['entropy'] = str(entropy_from_os)
            json.dump(inputs_json, json_file)

    output_list_full = [np.zeros((n_samples, len_beta, 3, 3)), # m_split
                     np.zeros((n_samples, len_beta, 3, 3)), # n_split
                     np.zeros((n_samples, len_beta), dtype=int), # ints_split
                     np.zeros((n_samples, len_beta, 3, 3)), # m_notsplit
                     np.zeros((n_samples, len_beta, 3, 3)), # n_notsplit
                     np.zeros((n_samples, len_beta), dtype=int) # ints_notsplit
                     ]

    for idx_s in range(n_samples):

        entropy = (entropy_from_os, idx_s)

        filenames = [
            file_npz[:-4] + f'_sample{idx_s}_m_split.npy',
            file_npz[:-4] + f'_sample{idx_s}_n_split.npy',
            file_npz[:-4] + f'_sample{idx_s}_ints_split.npy',
            file_npz[:-4] + f'_sample{idx_s}_m_notsplit.npy',
            file_npz[:-4] + f'_sample{idx_s}_n_notsplit.npy',
            file_npz[:-4] + f'_sample{idx_s}_ints_notsplit.npy'
        ]

        try:
            for idx_f, file in enumerate(filenames):
                output_list_full[idx_f][idx_s] = np.load(file)

        except FileNotFoundError:

            t = time()
            print(f'\nSolving system {idx_s + 1}/{n_samples}...')

            t0 = time()
            rng_seeds = np.random.SeedSequence(entropy).spawn(2)
            rng_seeds_split = rng_seeds[0].spawn(len_beta)
            rng_seeds_notsplit = rng_seeds[1].spawn(len_beta)
            print(f'Generated seeds in {round(time() - t0, 3)} s.')
            t0 = time()
            print('Generating systems...')

            split = hop(rngSS=rng_seeds[0], sigma_type='mix', noise_dif=True, **inputs_sys)

            inputs_sys_notsplit['K'] = split.pat

            jointex = np.full(shape=(split.L, inputs_sys_notsplit['M'], split.K, split.N),
                              fill_value=np.concatenate(tuple(layer for layer in split.ex)))

            notsplit = hop(rngSS=rng_seeds[1], sigma_type='mix', ex=jointex,
                           noise_dif=False, **inputs_sys_notsplit)
            print(f'Generated systems in {round(time() - t0, 3)} s.')

            def sim_func(beta, noise_split, noise_notsplit):
                output_m_split, output_n_split, ints_split = split.simulate(dynamic=dynamic, beta=beta,
                                                                            sim_rngSS=noise_split, cut=True,
                                                                            av=True, **inputs_sim)
                output_m_notsplit, output_n_notsplit, ints_notsplit = notsplit.simulate(dynamic=dynamic, beta=beta,
                                                                                        sim_rngSS=noise_notsplit,
                                                                                        cut=True, av=True,
                                                                                        **inputs_sim)
                return output_m_split, output_n_split, ints_split, output_m_notsplit, output_n_notsplit, ints_notsplit

            if parallel_CPUs:
                print(__name__) # DOES NOT WORK
                if __name__ == "__main__":
                    with Pool() as pool:
                        full_outputs = pool.starmap(sim_func, tqdm(zip(beta_values, rng_seeds_split, rng_seeds_notsplit), total = len_beta, disable = disable))
            else:
                full_outputs = [sim_func(beta, noise_split, noise_notsplit) for beta, noise_split, noise_notsplit in
                                tqdm(zip(beta_values, rng_seeds_split, rng_seeds_notsplit), total = len_beta, disable = disable)]

            print('Saving...')
            output_arrays = map(np.array, zip(*full_outputs))

            for idx_o, output in enumerate(output_arrays):
                np.save(filenames[idx_o], output)
                output_list_full[idx_o][idx_s] = output

            print(f'System ran in {round(time() - t / 60)} minutes.')

    return tuple(output_list_full)


def MCHop_InAndOut(neurons, K, rho, M, mixM, lmb, sigma_type, quality, noise_dif, beta, H, max_it, error, av_counter,
                   dynamic, L=3, h=None, rngSS=np.random.SeedSequence(), prints=False, cut=False):
    t = time()
    system = hop(neurons=neurons, L=L, K=K, rho=rho, M=M, mixM=mixM, lmb=lmb, sigma_type=sigma_type, quality=quality,
                 noise_dif=noise_dif, h=h, rngSS=rngSS)
    t = time() - t
    if prints:
        print(f'System generated in {round(t, 2)} secs.')

    sim_rngSS = rngSS.spawn(1)[0]

    return system.simulate(beta=beta, H=H, max_it=max_it, error=error, av_counter=av_counter,
                           dynamic=dynamic, cut=cut, disable=True, sim_rngSS=sim_rngSS)


def MC2d(directory, save_n, save_int, n_samples, y_values, y_arg, x_values, x_arg, dynamic, noise_dif, sigma_type,
         silent=False, disable=False, **kwargs):

    if silent:
        disable = True
    directory = directory
    len_y = len(y_values)
    len_x = len(x_values)

    json_dict = {'dynamic': dynamic,
                 'noise_dif': noise_dif,
                 'sigma_type': sigma_type,
                 'save_n': save_n,
                 'save_int': save_int}

    inputs_num = {**kwargs, x_arg: x_values, y_arg: y_values}

    inputs = {**json_dict, **kwargs, x_arg: x_values, y_arg: y_values}

    npz_files = npz_file_finder(directory=directory, prints=False, **inputs)
    if len(npz_files) > 1:
        print('Warning: more than 1 experiments found for given inputs.')

    try:
        file_npz = npz_files[0]
        with open(file_npz[:-3] + 'json', mode="r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            entropy_from_os = int(data['entropy'])
        if not silent:
            print('File found!')
            if n_samples > 0:
                print('Restarting...')
        samples_present = len([file for file in os.listdir(directory) if
                               file_npz[:-4] in os.path.join(directory, file) and file[-5:] == 'm.npy'])
        if not silent:
            print(f'There are {samples_present} sample(s) present')
        if n_samples == 0:
            if samples_present > 0:
                last_sample = np.load(file_npz[:-4] + f'_sample{samples_present - 1}_m.npy')
                if len(last_sample) < len_x * len_y:
                    samples_present -= 1
            if samples_present > 0:
                n_samples = samples_present
            else:
                raise Exception('No samples present. Compute some first.')

    except IndexError:
        print('No experiments found for given inputs. Starting one.')
        if n_samples == 0:
            raise Exception('No complete samples present. Compute some first.')
        file_npz = os.path.join(directory, f'MC2dF_{x_arg}{y_arg}{len_x * len_y}_{int(time())}.npz')
        entropy_from_os = np.random.SeedSequence().entropy
        with open(f'{file_npz[:-3]}json', mode="w", encoding="utf-8") as json_file:
            json_dict['entropy'] = str(entropy_from_os)
            json.dump(json_dict, json_file)
        np.savez(file_npz, **inputs_num)

    mattis = np.zeros((n_samples, len_x, len_y, 3, 3))
    mattis_ex = np.zeros((n_samples, len_x, len_y, 3, 3))
    max_ints = np.zeros((n_samples, len_x, len_y), dtype=int)

    inputs.pop('save_n')
    inputs.pop('save_int')

    for idx_s in range(n_samples):

        t0 = time()

        entropy = (entropy_from_os, idx_s)

        rng_seeds = np.random.SeedSequence(entropy).spawn(len_x * len_y)
        print(f'Generated seeds for simulate in {round(time() - t0, 3)} s.')

        t = time()
        if not silent:
            print(f'\nSolving system {idx_s + 1}/{n_samples}...')

        file_npy_m = file_npz[:-4] + f'_sample{idx_s}_m.npy'
        file_npy_n = file_npz[:-4] + f'_sample{idx_s}_n.npy'
        file_npy_ints = file_npz[:-4] + f'_sample{idx_s}_ints.npy'

        try:
            mattis_flat = np.load(file_npy_m)
            if save_n:
                mattis_flat_ex = np.load(file_npy_n)
                assert len(mattis_flat_ex) == len(mattis_flat), 'Sample files corrupted. Fix or delete.'
            else:
                mattis_flat_ex = []
            if save_int:
                flat_ints = np.load(file_npy_ints)
                assert len(flat_ints) == len(mattis_flat), 'Sample files corrupted. Fix or delete.'
            else:
                flat_ints = []

        except FileNotFoundError:
            mattis_flat = []
            mattis_flat_ex = []
            flat_ints = []

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
                        if save_int:
                            max_ints[idx_s, idx_x, idx_y] = flat_ints[idx_x * len_y + idx_y]
                    except IndexError:
                        inputs[y_arg] = y_v

                        output_m, output_n, ints = MCHop_InAndOut(cut=True, rngSS=rng_seeds[idx_x * len_y + idx_y],
                                                            **inputs)

                        output_m_mean = np.mean(output_m, axis=0)
                        output_n_mean = np.mean(output_n, axis=0)

                        mattis[idx_s, idx_x, idx_y] = output_m_mean
                        mattis_ex[idx_s, idx_x, idx_y] = output_n_mean
                        max_ints[idx_s, idx_x, idx_y] = ints

                        with NpyAppendArray(file_npy_m) as npyf:
                            npyf.append(output_m_mean.reshape((1, 3, 3)))

                        if save_n:
                            with NpyAppendArray(file_npy_n) as npyf:
                                npyf.append(output_n_mean.reshape((1, 3, 3)))
                        if save_int:
                            with NpyAppendArray(file_npy_ints) as npyf:
                                npyf.append(np.array([ints]))

                    if disable and not silent:
                        print(f'{x_arg} = {round(x_v, 2)}, {y_arg} = {round(y_v, 2)} ran to {max_ints[idx_s, idx_x, idx_y]} iteration(s).')
                    # print(f'MaxSD = {np.max(np.std(output, axis=0))}')
                    # print(f'MaxDif = {np.max(np.sum(np.diff(output, axis=0), axis=0))}')
                    else:
                        pbar.update(1)

        t = time() - t
        if not silent:
            print(f'System ran in {round(t / 60)} minutes.')

    return mattis, mattis_ex, max_ints


def MC2d_Lb(directory, save_n, save_int, n_samples, neurons, K, rho, M, mixM, lmb, dynamic, noise_dif, sigma_type, quality,
            disable=False,
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
                'mixM': mixM,
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

    npz_files = npz_file_finder(directory=directory, prints=False, **json_dict, **npz_dict, **sim_scalar_kwargs)

    if len(npz_files) > 1:
        print('Warning: more than 1 experiments found for given inputs.')

    try:
        file_npz = npz_files[0]
        with open(file_npz[:-3] + 'json', mode="r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            entropy_from_os = int(data['entropy'])
        print('File found. Restarting.')
        samples_present = len([file for file in os.listdir(directory) if
                               file_npz[:-4] in os.path.join(directory, file) and file[-5:] == 'm.npy'])
        print(f'There are {samples_present} sample(s) present')
        if n_samples == 0:
            if samples_present > 0:
                last_sample = np.load(file_npz[:-4] + f'_sample{samples_present - 1}_m.npy')
                if len(last_sample) < len_l * len_y:
                    samples_present -= 1
            if samples_present > 0:
                n_samples = samples_present
            else:
                raise Exception('No samples present. Compute some first.')

    except IndexError:
        print('No experiments found for given inputs. Starting one.')
        if n_samples == 0:
            raise Exception('No complete samples present. Compute some first.')
        file_npz = os.path.join(directory, f'MC2d_lmb{y_arg}{len_l * len_y}_{int(time())}.npz')
        entropy_from_os = np.random.SeedSequence().entropy
        with open(f'{file_npz[:-3]}json', mode="w", encoding="utf-8") as json_file:
            json_dict['entropy'] = str(entropy_from_os)
            json.dump(json_dict, json_file)
        np.savez(file_npz, **npz_dict, **sim_scalar_kwargs)

    mattis = np.zeros((n_samples, len_l, len_y, 3, 3))
    mattis_ex = np.zeros((n_samples, len_l, len_y, 3, 3))
    max_ints = np.zeros((n_samples, len_l, len_y))

    for idx_s in range(n_samples):
        t = time()
        print(f'\nSolving system {idx_s + 1}/{n_samples}...')

        file_npy_m = file_npz[:-4] + f'_sample{idx_s}_m.npy'
        file_npy_n = file_npz[:-4] + f'_sample{idx_s}_n.npy'
        file_npy_ints = file_npz[:-4] + f'_sample{idx_s}_ints.npy'

        try:
            mattis_flat = np.load(file_npy_m)
            if save_n:
                mattis_flat_ex = np.load(file_npy_n)
                assert len(mattis_flat_ex) == len(mattis_flat), 'Sample files corrupted. Fix or delete.'
            else:
                mattis_flat_ex = []
            if save_int:
                flat_ints = np.load(file_npy_ints)
                assert len(flat_ints) == len(mattis_flat), 'Sample files corrupted. Fix or delete.'
            else:
                flat_ints = []

        except FileNotFoundError:
            mattis_flat = []
            mattis_flat_ex = []
            flat_ints = []

        entropy = (entropy_from_os, idx_s)

        if len(mattis_flat) < len_l * len_y:
            if len(mattis_flat) == 0:
                print('Sample not present.')
            else:
                print(f'Sample incomplete ({len(mattis_flat)}/{len_l * len_y})')
            rngSS = np.random.SeedSequence(entropy)
            system = hop(neurons=neurons, K=K, L=3, rho=rho, M=M, noise_dif=noise_dif, sigma_type=sigma_type,
                         quality=quality, rngSS=rngSS, mixM=mixM)
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
                if len(mattis_flat) < (idx_l + 1) * len_y:
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
                        if save_int:
                            max_ints[idx_s, idx_l, idx_y] = flat_ints[idx_l * len_y + idx_y]
                    except IndexError:
                        new_inputs[y_arg] = y_v
                        output_m, output_n, ints = system.simulate(J=J_lmb, dynamic=dynamic, cut=True,
                                                             sim_rngSS=rng_seeds[idx_l * len_y + idx_y], **new_inputs)
                        output_m_mean = np.mean(output_m, axis=0)
                        output_n_mean = np.mean(output_m, axis=0)

                        mattis[idx_s, idx_l, idx_y] = output_m_mean
                        mattis_ex[idx_s, idx_l, idx_y] = output_n_mean
                        max_ints[idx_s, idx_l, idx_y] = ints

                        with NpyAppendArray(file_npy_m) as npyf:
                            npyf.append(output_m_mean.reshape((1, 3, 3)))
                        if save_n:
                            with NpyAppendArray(file_npy_n) as npyf:
                                npyf.append(output_n_mean.reshape((1, 3, 3)))
                        if save_int:
                            with NpyAppendArray(file_npy_ints) as npyf:
                                npyf.append(np.array([ints]))

                    if disable:
                        print(f'lmb = {round(lmb_v, 2)}, {y_arg} = {round(y_v, 2)} done.')
                    # print(f'MaxSD = {np.max(np.std(output, axis=0))}')
                    # print(f'MaxDif = {np.max(np.sum(np.diff(output, axis=0), axis=0))}')
                    else:
                        pbar.update(1)

        t = time() - t
        print(f'System ran in {round(t / 60)} minutes.')

    return mattis, mattis_ex, max_ints


def pat_id(m, cutoff_rec, cutoff_mix):
    for idx, mag in enumerate(m):
        if mag > cutoff_rec:
            return idx
        if mag < -cutoff_rec:
            return -idx
    if np.all(1 / 2 - cutoff_mix < m) and np.all(m < 1 / 2 + cutoff_mix):
        return 'mix'
    if np.all(1 / 2 - cutoff_mix < np.abs(m)) and np.all(np.abs(m) < 1 / 2 + cutoff_mix):
        return 'mix_signed'
    return None


def mags_id_old(m, cutoff_rec, cutoff_mix):
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


def mags_id(state, m, cutoff):
    if state == 'dis':
        pats = np.argmax(np.abs(m), axis=1)
        pats_mags = np.array([np.abs(m)[idx, pats[idx]] for idx in range(len(m))])
        if len(set(pats)) == len(pats) and np.all(pats_mags > cutoff):
            return True
        else:
            return False
    elif state == 'mix':
        if np.all(m > cutoff):
            return True
        else:
            return False
    else:
        return False


def gridvec_toplot(ax, state, m_array, limx0, limx1, limy0, limy1, cutoff, aspect='auto',
                   interpolate='x', **kwargs):
    all_samples, len_x, len_y, *rest = np.shape(m_array)
    success_array = np.zeros((all_samples, len_x, len_y))

    t = time()

    for idx_s in range(all_samples):
        for idx_x in range(len_x):
            for idx_y in range(len_y):
                if mags_id(state, m_array[idx_s, idx_x, idx_y], cutoff):
                    success_array[idx_s, idx_x, idx_y] = 1

    success_av = np.average(success_array, axis=0)

    vec_for_imshow = np.transpose(np.flip(success_av, axis=-1))
    # vec_for_imshow = np.transpose(success_av)
    print(f'Calculated success rates in {time() - t} seconds.')

    input_str = '_'.join(
        [f'{key}{int(1000 * value)}' if not np.isinf(value) else f'{key}inf' for key, value in kwargs.items()])
    disname = f'discurve_{input_str}'
    mixname = f'mixcurve_{input_str}'

    cutoffname = f'magdiscurve_{input_str}_c{int(1000 * cutoff)}'
    filesfromM = [disname, mixname, cutoffname]

    colorsfromM = ['red', 'blue', 'black']
    stylesfromM = ['solid', 'solid', 'dashed']
    interp_funcs = []
    tr_lines = []

    for idx_f, file in enumerate(filesfromM):
        try:
            data = mathToPython(file, 'TransitionData')
            if interpolate == 'x':
                interpolator = make_interp_spline(*data)
                interp_funcs.append(interpolator)
                x_values_smooth = np.linspace(start=data[0, 0], stop=data[0, -1], num=500, endpoint=True)
                if idx_f == 2:
                    x_values_smooth = np.array([x for x in x_values_smooth if
                                                interp_funcs[1](x) < interpolator(x) < interp_funcs[0](x)])
                    try:
                        idx_zero = np.where(interpolator(x_values_smooth) <= 0)[0][0]
                        x_values_smooth = x_values_smooth[:idx_zero]
                    except IndexError:
                        pass
                tr_lines.append([x_values_smooth, interpolator(x_values_smooth)])
            elif interpolate == 'y':
                interpolator = make_interp_spline(*(data[::-1]))
                interp_funcs.append(interpolator)
                y_values_smooth = np.linspace(start=data[1, 0], stop=data[1, -1], num=500, endpoint=True)
                if idx_f == 2:
                    y_values_smooth = [y for y in y_values_smooth if
                                       interp_funcs[1](y) < interpolator(y) < interp_funcs[0](y)]
                tr_lines.append([interpolator(y_values_smooth), y_values_smooth])

            else:
                tr_lines.append(data)
        except FileNotFoundError:
            tr_lines.append([[], []])

    c = ax.imshow(vec_for_imshow, cmap='Greens', vmin=0, vmax=1, aspect=aspect, interpolation='nearest',
                  extent=[limx0, limx1, limy0, limy1])

    ax.set_xlim(limx0, limx1)
    ax.set_ylim(limy0, limy1)

    for idx_line, line in enumerate(tr_lines):
        if interpolate:
            ax.plot(*line, color=colorsfromM[idx_line], linestyle=stylesfromM[idx_line], linewidth=2.0)
        else:
            ax.scatter(*line, color=colorsfromM[idx_line])

    return c
