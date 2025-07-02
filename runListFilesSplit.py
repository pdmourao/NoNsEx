import os
import numpy as np
from storage import npz_file_finder
from FPfields import m_in
import json


directory = 'ToSplit_Or_NotToSplit_beta'
excluded = ['beta']

trial = {'neurons': 5000, 'K': 5, 'M': 50, 'quality': [1, 1, 1], 'mixM': 0, 'rho': 0.45, 'lmb': 0.07, 'max_it': 30, 'error': 0.005, 'av_counter': 3, 'H': 0, 'sigma_type': 'mix', 'dynamic': 'sequential'}

T_values = np.linspace(start = 0, stop = 0.2, num = 101, endpoint = True)
with np.errstate(divide='ignore'):
    beta_values = 1/T_values

for file in npz_file_finder(directory, beta = beta_values, **trial):
    print('\n' + file)
    file_json = file[:-3] + 'json'
    newfile_json = file[:-4] + 'new.json'
    n_samples = 0
    for npy_file in os.listdir(directory):
        os_npy_file = os.path.join(directory, npy_file)
        if file[:-4] in os_npy_file and '_m_split' in os_npy_file:
            n_samples += 1
    print(f'{n_samples} sample(s).')
    # new_data = {}
    with open(file_json, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        for key in data:
            # new_data[key] = data[key]
            # if key == 'save_n':
            #     new_data['save_int'] = False
            if key not in excluded:
                print(f'{key}: {data[key]}')

    # with open(newfile_json, mode="w", encoding="utf-8") as json_newfile:
        # json.dump(new_data, json_newfile)
    # with open(file_json, mode="w", encoding="utf-8") as json_file:
    #     json.dump(new_data, json_file)
    # with open(file_json, mode="r", encoding="utf-8") as json_file:
    #     print(json.load(json_file))
    # with open(oldfile_json, mode="r", encoding="utf-8") as json_oldfile:
    #     print(json.load(json_oldfile))
    with np.load(file) as data:
        for key in data:
            if key not in excluded:
                print(f'{key} = {data[key]}')




