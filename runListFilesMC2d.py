import numpy as np
from storage import file_finder
from FPfields import m_in

directory = 'MC2d'
excluded = []

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)
beta_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]

for file in file_finder(directory):
    new_inputs = {}
    with np.load(file) as data:
        print(file)
        for key in data:
            new_inputs[key] = data[key]
            if key == 'K':
                new_inputs['lmb'] = l_values
                new_inputs['beta'] = beta_values
            if key not in excluded:
                print(key)
                print(data[key])




