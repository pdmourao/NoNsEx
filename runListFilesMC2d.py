import os
import numpy as np
from storage import npz_file_finder
from FPfields import m_in
import json


directory = 'MC2d'
excluded = ['lmb','rho','H']

x_values = np.linspace(start = 0, stop = 0.3, num = 50, endpoint = False)
y_values = np.linspace(start = 0, stop = 0.5, num = 50)

for file in npz_file_finder(directory):
    print('\n' + file)
    file_json = file[:-3] + 'json'
    # oldfile_json = file[:-4] + 'old.json'
    n_samples = 0
    for npy_file in os.listdir(directory):
        os_npy_file = os.path.join(directory, npy_file)
        if file[:-4] in os_npy_file and '_m' in os_npy_file:
            n_samples += 1
    print(f'{n_samples} sample(s).')
    with open(file_json, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        # new_data = {}
        for key in data:
            # new_data[key] = data[key]
            # if key == 'save_n':
            #     new_data['save_int'] = False
            if key not in excluded:
                print(f'{key}: {data[key]}')
    # try:
    #     data.pop('save_int')
    # except KeyError:
    #     pass
    # print(new_data)
    # print(data)
    # with open(oldfile_json, mode="w", encoding="utf-8") as json_oldfile:
    #     json.dump(data, json_oldfile)
    # with open(file_json, mode="w", encoding="utf-8") as json_file:
    #     json.dump(new_data, json_file)
    # with open(file_json, mode="r", encoding="utf-8") as json_file:
    #     print(json.load(json_file))
    # with open(oldfile_json, mode="r", encoding="utf-8") as json_oldfile:
    #     print(json.load(json_oldfile))
    with np.load(file) as data:
        for key in data:
            if key not in excluded and False:
                print(f'{key} = {data[key]}')




