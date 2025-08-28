import numpy as np
from MCclasses import HopfieldMC as hop
from tqdm import tqdm
from time import time, process_time
import json
import os
from storage import npz_file_finder, mathToPython
from npy_append_array import NpyAppendArray
from scipy.interpolate import make_interp_spline

from MCclasses import TAM as tam


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

    # this function runs the splitting experiment with only varying temperature
    # see the HopfieldMC class for more details

    len_beta = len(beta_values)

    inputs_sys = {'neurons': neurons, 'K': K, 'M': M, 'quality': quality, 'mixM': mixM, 'rho': rho, 'lmb': lmb, 'sigma_type': sigma_type}
    inputs_sys_notsplit = dict(inputs_sys)
    inputs_sys_notsplit['M'] = 3*M
    inputs_sys_notsplit['rho'] = rho/3
    inputs_sim = {'max_it': max_it, 'error': error, 'av_counter': av_counter, 'H': H, 'dynamic': dynamic}

    output_list = [np.zeros((len_beta, 3, 3)), # m_split
                     np.zeros((len_beta, 3, 3)), # n_split
                     np.zeros(len_beta, dtype=int), # ints_split
                     np.zeros((len_beta, 3, 3)), # m_notsplit
                     np.zeros((len_beta, 3, 3)), # n_notsplit
                     np.zeros(len_beta, dtype=int) # ints_notsplit
                     ]

    t = process_time()

    t0 = process_time()

    # the rng seeds for simulate
    rng_seeds = np.random.SeedSequence(entropy).spawn(2)
    rng_seeds_split = rng_seeds[0].spawn(len_beta)
    rng_seeds_notsplit = rng_seeds[1].spawn(len_beta)

    if not disable:
        print(f'Generated seeds in {round(process_time() - t0, 3)} s.')
    t0 = process_time()

    if not disable:
        print('Generating systems...')

    # initialize the systems
    # first create the split system and then join the examples to create the notsplit system
    split = hop(rngSS=rng_seeds[0], noise_dif=True, **inputs_sys)
    inputs_sys_notsplit['K'] = split.pat
    jointex = np.full(shape=(split.L, inputs_sys_notsplit['M'], split.K, split.N), fill_value=np.concatenate(tuple(layer for layer in split.ex)))
    notsplit = hop(rngSS=rng_seeds[1], ex=jointex, noise_dif=False, **inputs_sys_notsplit)

    if not disable:
        print(f'Generated systems in {round(process_time() - t0, 3)} s.')

    # run across the betas
    for beta_idx, beta in enumerate(tqdm(beta_values, disable = disable)):
        output_list[0][beta_idx], output_list[1][beta_idx], output_list[2][beta_idx] = split.simulate(beta=beta,sim_rngSS=rng_seeds_split[beta_idx], cut=True, av=True, **inputs_sim)
        output_list[3][beta_idx], output_list[4][beta_idx], output_list[5][beta_idx] = notsplit.simulate(beta=beta, sim_rngSS=rng_seeds_notsplit[beta_idx], cut=True, av=True, **inputs_sim)
    if not disable:
        print(f'Sample ran in {round(process_time() - t / 60)} minutes.')

    return tuple(output_list)


def disentanglement(neurons, k, r, m, lmb, split, supervised, beta, h_norm, max_it, error, av_counter, dynamic, checker = None, layers = 3, rng_ss = None, av = True):


    system = tam(neurons=neurons, layers = layers, r=r, m=m, lmb=lmb, split = split, supervised = supervised, rng_ss = rng_ss)


    system.noise_patterns = np.random.default_rng(rng_ss)
    system.noise_examples = system.noise_patterns

    system.add_patterns(k)
    system.initial_state = system.mix()
    system.external_field = system.mix(0)

    print(np.array_equal(system.J, checker))

    return system.simulate(beta=beta, h_norm = h_norm, max_it=max_it, error=error, av_counter=av_counter, dynamic=dynamic, av = av)


def disentanglement_2d(y_values, y_arg, x_values, x_arg, entropy = None, disable=False, **kwargs):

    len_y = len(y_values)
    len_x = len(x_values)

    mattis = np.zeros((len_x, len_y, 3, 3))
    mattis_ex = np.zeros((len_x, len_y, 3, 3))
    max_its = np.zeros((len_x, len_y), dtype=int)

    t0 = time()

    rng_seeds = np.random.SeedSequence(entropy).spawn(len_x * len_y)

    with tqdm(total=len_x * len_y, disable=disable) as pbar:
        for idx_x, x_v in enumerate(x_values):
            kwargs[x_arg] = x_v
            for idx_y, y_v in enumerate(y_values):
                kwargs[y_arg] = y_v
                mattis[idx_x, idx_y], mattis_ex[idx_x, idx_y], max_its[idx_x, idx_y] = disentanglement(
                    rng_ss=rng_seeds[idx_x * len_y + idx_y], **kwargs)
                if not disable:
                    pbar.update(1)

    print(f'System ran in {round((time()-t0 )/ 60)} minutes.')

    return mattis, mattis_ex, max_its


# Disentanglement experiments in terms of lambda and beta
def disentanglement_lmb_beta(neurons, k, r, m, lmb, beta, dynamic, split, supervised, max_it, error, av_counter, h_norm,
                             entropy = None, disable=False, checker = None):

    # Get length of input arrays
    len_lmb = len(lmb)
    len_beta = len(beta)

    mattis = np.zeros((len_lmb, len_beta, 3, 3))
    if supervised:
        mattis_ex = np.zeros((len_lmb, len_beta, 3, 3))
    else: # no average across examples in unsupervised experiments
        mattis_ex = np.zeros((len_lmb, len_beta, m, 3, 3))
    max_its = np.zeros((len_lmb, len_beta))

    t = time()

    system = tam(neurons=neurons, layers=3, r=r, m=m, split = split, supervised = supervised, rng_ss = np.random.SeedSequence(entropy=entropy))
    system.noise_patterns = np.random.default_rng(system.fast_noise)
    system.noise_examples = system.noise_patterns

    system.add_patterns(k)
    system.initial_state = system.mix()
    system.external_field = system.mix(0)



    with tqdm(total=len_lmb * len_beta, disable=disable) as pbar:
        for idx_lmb, lmb_v in enumerate(lmb):
            matrix_J = system.insert_g(lmb_v)
            for idx_beta, beta_v in enumerate(beta):

                mattis[idx_lmb, idx_beta], mattis_ex[idx_lmb, idx_beta], max_its[idx_lmb, idx_beta] = system.simulate(beta = beta_v, max_it = max_it, dynamic = dynamic, error = error, av_counter = av_counter, h_norm = h_norm, sim_J = matrix_J)
                if checker is not None:
                    assert np.array_equal(mattis[idx_lmb, idx_beta], checker[idx_lmb, idx_beta]), "Check not cleared"
                if disable:
                    print(f'lmb = {round(lmb_v, 2)}, beta = {round(beta_v, 2)} done.')
                # print(f'MaxSD = {np.max(np.std(output, axis=0))}')
                # print(f'MaxDif = {np.max(np.sum(np.diff(output, axis=0), axis=0))}')
                else:
                    pbar.update(1)

    t = time() - t
    print(f'System ran in {round(t / 60)} minutes.')

    return mattis, mattis_ex, max_its


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
