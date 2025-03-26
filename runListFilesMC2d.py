import os

import numpy as np
from storage import npz_file_finder
from FPfields import m_in
import json

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)

y_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]

directory = 'MC2d_Lb'
excluded = ['lmb', 'beta']

kwargs = {}

for file in npz_file_finder(directory, rho = 0.05, **kwargs):
    print('\n' + file)
    file_json = file[:-3] + 'json'
    n_samples = 0
    for npy_file in os.listdir(directory):
        os_npy_file = os.path.join(directory, npy_file)
        if file[:-4] in os_npy_file and '_m' in os_npy_file:
            n_samples += 1
    print(f'{n_samples} sample(s).')
    with open(file_json, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        for key in data:
            if key not in excluded:
                print(f'{key}: {data[key]}')
    with np.load(file) as data:
        new_data = dict(data)
        new_data['mixM'] = 0
        for key in data:
            if key not in excluded:
                print(f'{key} = {data[key]}')
        # np.savez(file[:-4] + 'new.npz', **new_data)




