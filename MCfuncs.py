import numpy as np
from MCclasses import HopfieldMC as hop, HopfieldMC_rho as hop_rho
from tqdm import tqdm
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
# For example, high temperatures take a long time and we know they give 0
# pbar is for the progress bar (see runMCHopfield_Lbeta.py)

def disentangler(beta, cutoff, J, system = None, max_it = 30, error = 1e-3, H = 0, parallel = False, pbar = None, *args, **kwargs):

    # Case where systems are not provided a priori
    if system is None:
        system = hop(*args, **kwargs)

    T = 1 / beta

    success = 0

    final_sigma = system.simulate(max_it = max_it, error = error, J = J, T = T, H = H, parallel = parallel)[-1]
    m = system.mattis(final_sigma)


    return m


# IsDisentangled checks whether an L x L matrix is disentangled
# This is done by checking every line of the matrix for entries > cutoff
# And then checking if the indices of those entries are all different (thanks Andrea)

def recovered_pats(m, cutoff):
    m_entries = [None, None, None]
    for idx1, m1 in enumerate(m):
        for idx2, m2 in enumerate(m1):
            if m2 > cutoff:
                m_entries[idx2] = idx1+1
            elif m2 < - cutoff:
                m_entries[idx2] = -idx1-1
    return tuple(m_entries)


# gJprod inserts a g matrix into an already computed J
# (see in HopfieldMC class why these are separated)

def gJprod(g, J):
    return np.transpose(np.transpose(J, [1, 3, 0, 2]) * g, [2, 0, 3, 1])

def MC2d_Lb(neurons, K, rho, M, H, l_values, beta_values, max_it, error, parallel, disable = False):
    system = hop(N=neurons, pat=K, L=3, quality=(rho, M))

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

                mattisses[idx_l, idx_b] = system.mattis(
                    system.simulate(max_it=max_it, error = error, J=J_lmb, T=1 / beta, H=H,
                                    parallel=parallel)[-1])
                if pbar is not None:
                    pbar.update(1)

    return mattisses


def MCrho(neurons, K, lmb, rho_values, M, error, beta, H, max_it, disable = False, parallel = False):

    m_array_initial = np.zeros(shape=(len(rho_values), 3, 3))
    m_array = np.zeros(shape=(len(rho_values), 3, 3))

    g = np.array([[1, - lmb, - lmb],
                  [- lmb, 1, - lmb],
                  [- lmb, - lmb, 1]])

    with tqdm(total=len(rho_values), disable=disable) as pbar:
        for idx_rho, rho in enumerate(rho_values):
            system = hop(N=neurons, pat=K, L=3, quality=(rho, M))
            m_array_initial[idx_rho] = system.mattis(system.sigma)
            last_sigma = system.simulate(max_it=max_it, error=error, J=gJprod(g, system.J), T=1 / beta, H=H, parallel=parallel)[-1]
            m_array[idx_rho] = system.mattis(last_sigma)
            pbar.update(1)

    return m_array_initial, m_array


def MCrho_s(neurons, K, rho_values, M, error, beta, max_it, disable = False, parallel = False):

    m_array_initial = np.zeros(shape=(len(rho_values), K))
    m_array = np.zeros(shape=(len(rho_values), K))

    with tqdm(total=len(rho_values), disable=disable) as pbar:
        for idx_rho, rho in enumerate(rho_values):
            print(rho)
            system = hop_rho(N=neurons, K=K, L=3, quality=(rho, M), beta = beta)
            # system2 = hop(N = neurons, L = 3, pat = system.pats, ex = np.full(shape = (3, M, K, neurons), fill_value = system.ex), quality = (rho, M))

            m_array_initial[idx_rho] = system.mattis(system.sigma)
            last_sigma = system.simulate(max_it=max_it, error = error, parallel=parallel)
            m_array[idx_rho] = system.mattis(last_sigma)
            pbar.update(1)

    return m_array_initial, m_array