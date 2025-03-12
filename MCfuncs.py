import numpy as np
from MCclasses import HopfieldMC as hop, HopfieldMC_rho as hop_rho
from MCclasses_tf import HopfieldMC_tf as hop_tf
from tqdm import tqdm
from time import time

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
# pbar is for the progress bar (see runMCHopfield_Lbeta.py)






# gJprod inserts a g matrix into an already computed J
# (see in HopfieldMC class why these are separated)

def gJprod(g, J):
    return np.transpose(np.transpose(J, [1, 3, 0, 2]) * g, [2, 0, 3, 1])


def MC2d_Lb(neurons, K, rho, M, H, l_values, beta_values, max_it, error, parallel, noise_dif, sigma_type, use_tf = False,
            av_counter = 10, disable = False):

    if parallel and use_tf:
        system = hop_tf(N=neurons, pat=K, L=3, rho = rho, M = M, noise_dif=noise_dif, sigma_type = sigma_type)
    else:
        system = hop(N=neurons, pat=K, L=3, rho = rho, M = M, noise_dif=noise_dif, sigma_type = sigma_type)

    len_l = len(l_values)
    len_b = len(beta_values)
    mattisses = np.zeros(shape=(len_l, len_b, 3, 3))

    with tqdm(total=len_l*len_b, disable=disable) as pbar:

        for idx_l, lmb in enumerate(l_values):
            g = np.array([[1, - lmb, - lmb],
                          [- lmb, 1, - lmb],
                          [- lmb, - lmb, 1]])
            J_lmb = gJprod(g, system.J)

            for idx_b, beta in enumerate(beta_values):
                output = np.array(system.simulate(av_counter=av_counter, error=error, J=J_lmb, beta = beta, H=H,
                                                  parallel=parallel, max_it = max_it, cut = True)[0])

                mattisses[idx_l, idx_b] = np.mean(output, axis = 0)

                if disable:
                    print(f'lmb = {round(lmb, 2)}, b = {round(beta, 2)} done.')
                    print(f'MaxSD = {np.max(np.std(output, axis = 0))}')
                    print(f'MaxDif = {np.max(np.sum(np.diff(output, axis = 0), axis=0))}')
                else:
                    pbar.update(1)

    return mattisses

def MC1d_beta(neurons, K, rho, M, H, lmb, beta, max_it, error, quality, parallel, noise_dif, random_systems = True, use_tf = False,
            av_counter = 10, disable = False):

    mattisses = np.zeros(shape=(len(beta), 3, 3))

    if random_systems:
        print('Generating systems...')
        if parallel and use_tf:
            systems = [hop_tf(L=3, noise_dif=noise_dif, N = neurons, pat = K, rho = rho, M = M, sigma_type = 'dis', sigma_quality = quality) for _ in tqdm(beta)]
        else:
            systems = [hop(L=3, noise_dif=noise_dif, N = neurons, pat = K, lmb = lmb, rho = rho, M = M, sigma_type = 'dis', sigma_quality = quality) for _ in tqdm(beta)]
    else:
        print('Generating system...')
        if parallel and use_tf:
            systems = hop_tf(L=3, noise_dif=noise_dif, N = neurons, pat = K, rho = rho, M = M, sigma_type = 'dis', sigma_quality = quality)
        else:
            systems = hop(L=3, noise_dif=noise_dif, N = neurons, pat = K, lmb = lmb, rho = rho, M = M, sigma_type = 'dis', sigma_quality = quality)

    for idx_b, beta_value in enumerate(tqdm(beta, disable=disable)):
        t = time()
        if random_systems:
            system = systems[idx_b]
        else:
            system = systems

        output = np.array(system.simulate(beta=beta_value, parallel=parallel, cut=False, H=H, max_it=max_it, error=error,
                                          av_counter=av_counter)[0])

        mattisses[idx_b] = np.mean(output[-av_counter:], axis=0)

        if disable:
            print(f'\nT = {round(1/beta_value, 2)} done.')
            print(f'Output after {len(output)-1} iterations ({round(time() - t, 2)}s):')
            print(mattisses[idx_b])

    return mattisses

# THIS IS NOT UP TO DATE
def MCrho(neurons, K, lmb, rho_values, M, error, beta, H, max_it, av_counter, disable = False, parallel = False):

    m_array_initial = np.zeros(shape=(len(rho_values), 3, 3))
    m_array = np.zeros(shape=(len(rho_values), 3, 3))

    g = np.array([[1, - lmb, - lmb],
                  [- lmb, 1, - lmb],
                  [- lmb, - lmb, 1]])

    with tqdm(total=len(rho_values), disable=disable) as pbar:
        for idx_rho, rho in enumerate(rho_values):
            system = hop(N=neurons, pat=K, L=3, rho = rho, M = M)
            m_array_initial[idx_rho] = system.mattis(system.sigma)
            last_sigma = system.simulate(max_it=max_it, error=error, J=gJprod(g, system.J), beta = beta, H=H,
                                         parallel=parallel, av_counter = av_counter)[-1]
            m_array[idx_rho] = system.mattis(last_sigma)
            pbar.update(1)

    return m_array_initial, m_array


def MCrho_s(neurons, K, rho_values, M, error, beta, max_it, disable = False, parallel = False):

    m_array_initial = np.zeros(shape=(len(rho_values), K))
    m_array = np.zeros(shape=(len(rho_values), K))

    with tqdm(total=len(rho_values), disable=disable) as pbar:
        for idx_rho, rho in enumerate(rho_values):
            print(rho)
            system = hop_rho(N=neurons, pat=K, L=3, rho = rho, M = M)
            # system2 = hop(N = neurons, L = 3, pat = system.pats, ex = np.full(shape = (3, M, K, neurons), fill_value = system.ex), quality = (rho, M))

            m_array_initial[idx_rho] = system.mattis(system.sigma)
            last_sigma = system.simulate(beta = beta, max_it=max_it, error = error, parallel=parallel)
            m_array[idx_rho] = system.mattis(last_sigma)
            pbar.update(1)

    return m_array_initial, m_array