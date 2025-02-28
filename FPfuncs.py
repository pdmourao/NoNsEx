from time import time
import numpy as np
from tqdm import tqdm
import os
from storage import file_finder
from copy import deepcopy

def iterator(*args, max_it, field, not_all_neg = [], error = 0, order = None, ibound = 0, min_it = 1, pbar = None, **kwargs):
    iter_list = [args]

    for it in range(max_it):

        old_iter = iter_list[-1]

        # calculates new entry
        new_iter = field(*old_iter, errorbound = ibound, **kwargs)
        # print(new_n)
        # print(new_q)
        # calculate the error
        # if the difference is not smaller than error, add it to history, otherwise break

        if error > 0:
            difs = [np.linalg.norm(new_iter[idx] - old_iter[idx], ord = order) for idx in range(len(args))]

            # this part is just if we want to avoid results oscillating between something and its symmetric
            # this is only checked for the list of indices not_all_neg
            changeable_iter = list(new_iter)
            for reversable_idx in not_all_neg:
                reversed_dif = np.linalg.norm(new_iter[reversable_idx] + old_iter[reversable_idx], ord = order)
                if reversed_dif < difs[reversable_idx]:
                    difs[reversable_idx] = reversed_dif
                    changeable_iter[reversable_idx] = - changeable_iter[reversable_idx]
            if it >= min_it and sum(difs) < error:
                break
            else:
                iter_list.append(tuple(changeable_iter))
    if pbar is not None:
        pbar.update(1)
    return iter_list


def solve(field, *args, rand = None, use_files = False, **kwargs):

    x_arg = None
    x_values = None

    if rand is None:
        directory = 'FP1d'
        idc = ''
        new_args = args
    else:
        directory = 'FP1drand'
        idc = f'_r{rand[0]}'
        new_args = list(args)
        with np.load('rand_mags.npz') as data:
            pert = data[f'arr_{rand[0]}']
        new_args[0] = args[0] + rand[1]*pert
        print('Perturbation matrix used:')
        print(rand[1]*pert)


    for key, value in kwargs.items():
        if not np.isscalar(value):
            if x_arg is not None:
                print('Warning: multiple arrays given as inputs.')
            x_arg = key
            x_values = value
            filename = os.path.join(directory, f'{field.__name__}_{x_arg}{len(x_arg)}{idc}_{int(time())}')

    if use_files:
        try:
            with np.load(file_finder(directory, *new_args, **kwargs)[0]) as data:
                m = data['m']
                q = data['q']
                n = data['n']
            return m, q, n
        except IndexError:
            pass

    if x_arg is None:
        t = time()
        values_list = iterator(*new_args, field=field, **kwargs)
        print(f'0D Iterator ran in {round(time() - t, 2)} seconds')
        m = np.zeros((len(values_list) - 1, 3, 3))
        n = np.zeros((len(values_list) - 1, 3, 3))
        q = np.zeros((len(values_list) - 1, 3))
        for idx, m_entry, q_entry, n_entry in enumerate(values_list[1:]):
            m[idx] = m_entry
            n[idx] = n_entry
            q[idx] = q_entry
    else:
        m = np.zeros((len(x_values), 3, 3))
        n = np.zeros((len(x_values), 3, 3))
        q = np.zeros((len(x_values), 3))
        t = time()
        for idx, value in enumerate(tqdm(x_values, disable=False)):
            kwargs[x_arg] = value
            m[idx], q[idx], n[idx] = iterator(*new_args, field=field, **kwargs)[-1]
        print(f'1D Iterator ran in {round((time() - t)/60, 2)} minutes')
        kwargs[x_arg] = x_values
        if use_files:
            print(f'Saving as {filename} :)')
            np.savez(filename, *new_args, m=m, n=n, q=q, **kwargs)



    return m, q, n


def FindTransition(vec_m, tr_det, **kwargs):

    tr_idx = -1

    for idx, m in enumerate(vec_m):
        if tr_det(vec_m[idx], **kwargs):
            tr_idx = idx
            break
    return tr_idx


# Example of a transition detector
# it should have two inputs
# these are meant to be the result at the previous value of x and at the current one
# should return True or False
# Just add conditions as necessary
def tr_det_NoNsEx(value_new, threshold = 1e-5):
    if np.var(np.diag(value_new)) > threshold:
        return True
    else:
        return False


def disentangle_det(value_new, threshold):
    m_entries = []
    for m1 in value_new:
        for idx2, m2 in enumerate(m1):
            if m2 > threshold:
                m_entries.append(idx2)
    if len(set(m_entries)) == 3:
        return True
    else:
        return False

def tr_notdis_NoNsEx(value_new, threshold1, threshold2 = 1e-5):
    if not disentangle_det(value_new, threshold1) and (tr_det_NoNsEx(value_new, threshold2) or disentangle_det(value_new, 0.6)):
        return True
    else:
        return False


def thresh_NoNsEx(value_new, threshold):
    verdict = False
    if np.mean(np.diag(value_new)) < threshold:
        verdict = True
    return verdict


# To use for axis labels
arg_to_label = {'initial_m': 'ε', 'it': 'it', 'beta': 'β', 'rho': 'ρ', 'lmb': 'λ', 'T': 'T', 'alpha': 'α', 'H': 'H'}
