import numpy as np
from storage import npz_file_finder
from FPfields import m_in
import json

directory = 'MC2d_Lb'
excluded = ['lmb', 'beta', 'H', 'entropy']

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)
beta_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]

for file in npz_file_finder(directory):
    print('\n' + file)
    file_json = file[:-3] + 'json'
    with open(file_json, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        for key in data:
            if key not in excluded:
                print(f'{key}: {data[key]}')
    with np.load(file) as data:
        for key in data:
            if key not in excluded:
                print(f'{key} = {data[key]}')




