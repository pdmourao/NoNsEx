import os
import numpy as np
from storage import npz_file_finder, exp_finder
from FPfields import m_in
import json

# be careful running this code, its meant to copy to 'MCData'
# take care of -14 and other things first (maybe not copy all etc)
# save lines have been commented
directory = 'ToSplit_Or_NotToSplit_beta'
excluded = ['beta']

trial = {'neurons': 5000, 'K': 5, 'M': 50, 'quality': [1, 1, 1], 'mixM': 0, 'rho': 0.45, 'lmb': 0.07, 'max_it': 30, 'error': 0.005, 'av_counter': 3, 'H': 0, 'sigma_type': 'mix', 'dynamic': 'sequential'}

T_values = np.linspace(start = 0, stop = 0.2, num = 101, endpoint = True)
with np.errstate(divide='ignore'):
    beta_values = 1/T_values

for file in npz_file_finder(directory):
    print('\n' + file)
    file_json = file[:-3] + 'json'
    newfile_json = os.path.join('MCData','splitting_beta-' + file[-14:-4] + '_inputs.json')
    newfile_npz = os.path.join('MCData', 'splitting_beta-' + file[-14:-4] + '_inputs.npz')
    print('New files:')
    print(newfile_json)
    print(newfile_npz)
    n_samples = 0
    for npy_file in os.listdir(directory):
        os_npy_file = os.path.join(directory, npy_file)
        if file[:-4] in os_npy_file and '_m_split' in os_npy_file:
            sample = int(os_npy_file.split('_')[-3][6:])
            new_output = (np.load(os_npy_file),
                          np.load(os_npy_file.replace('m_split', 'n_split')),
                          np.load(os_npy_file.replace('m_split', 'ints_split')),
                          np.load(os_npy_file.replace('m_split', 'm_notsplit')),
                          np.load(os_npy_file.replace('m_split', 'n_notsplit')),
                          np.load(os_npy_file.replace('m_split', 'ints_notsplit'))
                          )
            new_output_file = newfile_npz.replace('inputs', f'sample{sample}')
            # np.savez(new_output_file, *new_output)
            print(new_output_file)
            n_samples += 1
    print(f'{n_samples} sample(s).')
    # new_data = {}
    with open(file_json, mode="r", encoding="utf-8") as json_file:
        data_json = json.load(json_file)
        for key in data_json:
            if key not in excluded:
                print(f'{key}: {data_json[key]}')
    # with open(newfile_json, mode="w", encoding="utf-8") as json_file:
    #     json.dump(data_json, json_file)

    with np.load(file) as data_npz:
        new_data_npz = dict(data_npz)
        for key, value in data_npz.items():
            if key == 'beta':
                new_data_npz['beta_values'] = value
                new_data_npz.pop('beta')
            if key not in excluded:
                print(f'{key} = {value}')
    # np.savez(newfile_npz, **new_data_npz)




