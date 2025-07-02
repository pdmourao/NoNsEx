from time import time
import numpy as np
from tqdm import tqdm
import os
from storage import npz_file_finder
from multiprocessing import Pool

#comment
def iterator(*args, max_it, field, not_all_neg = [], error = 0, order = None, min_it = 1,
             pbar = None, **kwargs):

    new_iter = args
    it = 0

    while it < max_it:
        it += 1
        old_iter = new_iter
        # calculates new entry
        new_iter = field(*old_iter, **kwargs)

        # calculate differences
        difs = [np.linalg.norm(new_iter[idx] - old_iter[idx], ord=order) for idx in range(len(old_iter))]
        # this part is just if we want to avoid results oscillating between something and its symmetric
        # this is only checked for the list of indices not_all_neg
        if len(not_all_neg) > 0:
            changeable_iter = list(new_iter)
            for reversable_idx in not_all_neg:
                reversed_dif = np.linalg.norm(new_iter[reversable_idx] + old_iter[reversable_idx], ord=order)
                if reversed_dif < difs[reversable_idx]:
                    difs[reversable_idx] = reversed_dif
                    changeable_iter[reversable_idx] = - changeable_iter[reversable_idx]
            new_iter = tuple(changeable_iter)
        print(f'Iteration {it}:')
        print(new_iter)
        print('Error:')
        print(difs)
        # print(f'error = {difs[0]}')
        if it >= min_it and sum(difs) < error:
            break

    if pbar is not None:
        pbar.update(1)

    return it, *new_iter

def solve(field, *args, x_arg = None, directory = None, disable = False, parallel_CPUs = False, **kwargs):


    # try to get results already computed from files
    try:
        file_list_npz = npz_file_finder(directory, *args, **kwargs)
        fname_os = file_list_npz[0]
        if len(file_list_npz) > 0:
            print('Warning: More than one file found.')
            print(f'Using {fname_os}')
        file_list_npy = [os.path.join(directory, file) for file in os.listdir(directory) if fname_os[:-4] in os.path.join(directory, file)]
        file_list_npy.sort()
        outputs = [np.load(file) for file in file_list_npy]
        output_tuple = tuple(outputs)
    except (IndexError, TypeError) as e:
        t = time()
        # 0D solver (array not given for any of the arguments)
        if x_arg is None:
            output_tuple = iterator(*args, field=field, **kwargs)
            print(f'0D Iterator ran in {round((time() - t) / 60, 2)} minutes')
            print(f'Ran to {output_tuple[0]} iterations.')
            print('Output:')
            [print(output) for output in output_tuple]

        # 1D solver (array given for one of the arguments)
        else:
            x_values = kwargs[x_arg]
            output_list = []

            fixed_kwargs = dict(kwargs)
            fixed_kwargs.pop(x_arg)
            if disable:
                pbar = None
            else:
                pbar = tqdm(total=len(x_values))

            # the version of the iterator used for these parameters
            def this_iterator(x):
                t0 = time()
                output = iterator(*args, field=field, pbar=pbar, **{x_arg: x}, **fixed_kwargs)
                if disable:
                    print(f'{x_arg} = {x} solved in {round(time() - t0)} seconds.')
                    print(f'Ran to {output[0]} iterations.')
                    print('Output:')
                    [print(out) for out in output[1:]]
                return output

            if parallel_CPUs:
                if __name__ == "__main__":
                    with Pool() as pool:
                        output_list = pool.map(this_iterator, x_values)
            else:
                output_list = [this_iterator(value) for value in x_values]

            if not disable:
                pbar.close()
            print(f'1D Iterator ran in {round((time() - t)/60, 2)} minutes')
            # turns a list of tuples into a tuple of np.arrays
            output_tuple = tuple(map(np.array, zip(*output_list)))
        # tries to save to files
        try:
            handle = f'{field.__name__}_{int(time())}'
            filename = os.path.join(directory, handle)
            for idx_o, output in enumerate(output_tuple):
                np.save(filename + f'_output{idx_o}.npy', output)
            np.savez(filename, *args, **kwargs)
            print(f'Saved as {handle} :)')
        except TypeError:
            print('Results not saved.')

    return output_tuple

def solve_old(field, *args, rand = None, use_files = False, disable = False, **kwargs):

    x_arg = None
    x_values = None

    if rand is None:
        directory = 'FP1d_old'
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
            filename = os.path.join(directory, f'{field.__name__}_{x_arg}{len(x_values)}{idc}_{int(time())}')

    if use_files:
        try:
            with np.load(npz_file_finder(directory, *new_args, **kwargs)[0]) as data:
                m = data['m']
                q = data['q']
                n = data['n']
            return m, q, n
        except IndexError:
            pass

    t = time()
    values_list = []
    for idx, value in enumerate(tqdm(x_values, disable=disable)):
        t0 = time()
        kwargs[x_arg] = value
        maxed_it, output = iterator(*new_args, field=field, **kwargs)
        if disable:
            print(f'{x_arg} = {value} solved in {round(time() - t0)} seconds.')
            print(f'Ran to {maxed_it+1} iterations.')
            print('Output:')
            print(output[-1])
        values_list.append(output[-1])
    print(f'1D Iterator ran in {round((time() - t)/60, 2)} minutes')
    kwargs[x_arg] = x_values
    output_tuple = tuple(map(np.array, zip(*values_list)))
    if use_files and x_arg is not None:
        output_args = ('m', 'q', 'n')
        output_dict = {arg: output_tuple[idx] for idx, arg in enumerate(output_args)}
        print(f'Saving as {filename} :)')
        np.savez(filename, *new_args, **output_dict, **kwargs)

    return output_tuple


def FindTransitionFromVec(vec_m, tr_det, **kwargs):

    tr_idx = None

    for idx, m in enumerate(vec_m):
        if tr_det(vec_m[idx], **kwargs):
            tr_idx = idx
            break
    return tr_idx

def FindTransitionL(field, ):
    return None


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


# recovered_pats checks what pattern has been recovered at each layer or if it is a mixture

def recovered_pats(m, cutoff_recovered, cutoff_mix = 0.1):
    m_entries = [None, None, None]
    for idx1, m1 in enumerate(m):
        for idx2, m2 in enumerate(m1):
            if m2 > cutoff_recovered:
                m_entries[idx1] = idx2+1
            elif m2 < - cutoff_recovered:
                m_entries[idx1] = -idx2-1
        if all([1/2 - cutoff_mix < m2 < 1/2 + cutoff_mix for m2 in m1]):
            if m_entries[idx1] is not None:
                print('Both recovered and mixed ?!')
            m_entries[idx1] = 4
    return tuple(m_entries)


# To use for axis labels
arg_to_label = {'initial_m': 'ε', 'it': 'it', 'beta': 'β', 'rho': 'ρ', 'lmb': 'λ', 'T': 'T', 'alpha': 'α', 'H': 'H'}
