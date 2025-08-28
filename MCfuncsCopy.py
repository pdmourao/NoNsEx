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


def splitting_optimal(entropy, neurons, layers, k, m, rho_values, suf,  max_it, error, av_counter, h_norm, minlmb, minT, dynamic, disable = False):

    t = process_time()
    len_rho = len(rho_values)

    interpolatorT = make_interp_spline(*mathToPython('maxT'+suf,'optpar'))
    interpolatorL = make_interp_spline(*mathToPython('maxL'+suf,'optpar'))


    mattis_split = np.zeros((len_rho, 3, 3))
    mattis_ex_split = np.zeros((len_rho, 3, 3))
    max_its_split = np.zeros((len_rho), dtype=int)
    mattis_notsplit = np.zeros((len_rho, 3, 3))
    mattis_ex_notsplit = np.zeros((len_rho, 3, 3))
    max_its_notsplit = np.zeros((len_rho), dtype=int)

    rng_seeds = np.random.SeedSequence(entropy).spawn(len_rho)

    with tqdm(total=len_rho, disable=disable) as pbar:
        for idx_rho, rho_v in enumerate(rho_values):

            lmb_split = max(interpolatorL(3*rho_v), minlmb)
            lmb_notsplit = max(interpolatorL(rho_v), minlmb)

            beta_split = 1 / interpolatorT(3*rho_v) if interpolatorT(3*rho_v) > minT else np.inf
            beta_notsplit = 1 / interpolatorT(rho_v) if interpolatorT(rho_v) > minT else np.inf

            r = np.sqrt(1 / (rho_v * m + 1))

            noise_split = rng_seeds[2 * idx_rho]
            noise_notsplit = rng_seeds[2 * idx_rho + 1]

            system = tam(neurons = neurons, layers = layers, r = r, m = m, supervised = True, rng_ss=noise_split, lmb=lmb_split,
                         split = True)

            system.noise_patterns = np.random.default_rng(noise_split)
            system.noise_examples = system.noise_patterns

            system.add_patterns(k)

            mattis_split[idx_rho], mattis_ex_split[idx_rho], max_its_split[idx_rho] = system.simulate(beta=beta_split, dynamic=dynamic, av_counter = av_counter, h_norm = h_norm, max_it = max_it, error = error, sim_rng = system.fast_noise)

            system.set_interaction(lmb = lmb_notsplit, split = False)
            system.fast_noise = noise_notsplit

            mattis_notsplit[idx_rho], mattis_ex_notsplit[idx_rho], max_its_notsplit[idx_rho] = system.simulate(beta=beta_notsplit, dynamic=dynamic, av_counter = av_counter, h_norm = h_norm, max_it = max_it, error = error)


            if not disable:
                pbar.update(1)

    t = time() - t
    print(f'System ran in {round(t / 60)} minutes.')

    return mattis_split, mattis_ex_split, max_its_split, mattis_notsplit, mattis_ex_notsplit, max_its_notsplit


def splitting_beta(entropy, beta_values, neurons, k, layers, supervised, r, lmb, m, max_it, error, av_counter, h_norm, dynamic, disable = True):

    # this function runs the splitting experiment with only varying temperature
    # see the HopfieldMC class for more details

    len_beta = len(beta_values)

    mattis_split = np.zeros((len_beta, 3, 3))
    mattis_ex_split = np.zeros((len_beta, 3, 3))
    max_its_split = np.zeros((len_beta), dtype=int)
    mattis_notsplit = np.zeros((len_beta, 3, 3))
    mattis_ex_notsplit = np.zeros((len_beta, 3, 3))
    max_its_notsplit = np.zeros((len_beta), dtype=int)

    t = process_time()

    # the rng seeds for simulate
    rng_seeds = np.random.SeedSequence(entropy).spawn(2)
    rng_seeds_split = rng_seeds[0].spawn(len_beta)
    rng_seeds_notsplit = rng_seeds[1].spawn(len_beta)


    system = tam(neurons = neurons, r = r, lmb = lmb, m = m, supervised = supervised, split = True, layers = layers)
    system.noise_patterns = np.random.default_rng(rng_seeds[0])
    system.noise_examples = system.noise_patterns

    system.add_patterns(k)

    # run across the betas
    for beta_idx, beta in enumerate(tqdm(beta_values, disable = disable)):

        mattis_split[beta_idx], mattis_ex_split[beta_idx], max_its_split[beta_idx] = system.simulate(beta=beta, sim_rng=rng_seeds_split[beta_idx], av_counter = av_counter, max_it = max_it, error = error, h_norm = h_norm, dynamic = dynamic)
        system.set_interaction(split = False)
        mattis_notsplit[3][beta_idx], mattis_ex_notsplit[4][beta_idx], max_its_notsplit[beta_idx] = system.simulate(beta=beta, sim_rng=rng_seeds_notsplit[beta_idx], av_counter = av_counter, max_it = max_it, error = error, h_norm = h_norm, dynamic = dynamic)

    if not disable:
        print(f'Sample ran in {round(process_time() - t / 60)} minutes.')

    return mattis_split, mattis_ex_split, max_its_split, mattis_notsplit, mattis_ex_notsplit, max_its_notsplit


def disentanglement(neurons, layers, k, r, m, lmb, split, supervised, beta, h_norm, max_it, error, av_counter, dynamic, rng_ss = None, av = True):

    system = tam(neurons=neurons, layers = layers, r=r, m=m, lmb=lmb, split = split, supervised = supervised)

    system.noise_patterns = np.random.default_rng(rng_ss)
    system.noise_examples = system.noise_patterns
    system.fast_noise = rng_ss

    system.add_patterns(k)
    system.initial_state = system.mix()
    system.external_field = system.mix(0)


    return system.simulate(beta=beta, h_norm = h_norm, max_it=max_it, error=error, av_counter=av_counter, dynamic=dynamic, av = av)


def disentanglement_2d(y_values, y_arg, x_values, x_arg, entropy = None, disable=False, checker = None, **kwargs):

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
                mattis[idx_x, idx_y], mattis_ex[idx_x, idx_y], max_its[idx_x, idx_y] = disentanglement(rng_ss=rng_seeds[idx_x * len_y + idx_y], **kwargs)
                assert np.array_equal(mattis[idx_x, idx_y], checker[idx_x, idx_y]), 'Check 1 didnt pass'
                if not disable:
                    pbar.update(1)

    print(f'System ran in {round((time()-t0 )/ 60)} minutes.')

    return mattis, mattis_ex, max_its


# Disentanglement experiments in terms of lambda and beta
def disentanglement_lmb_beta(neurons, layers, k, r, m, lmb, beta, dynamic, split, supervised, max_it, error, av_counter, h_norm,
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

    system = tam(neurons=neurons, layers = layers, r=r, m=m, split = split, supervised = supervised)
    system.noise_patterns = np.random.default_rng(np.random.SeedSequence(entropy=entropy))
    system.noise_examples = system.noise_patterns
    system.fast_noise = np.random.SeedSequence(entropy=entropy)

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
