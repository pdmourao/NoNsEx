import numpy as np
from storage import file_finder

directory = 'FP1d'
excluded = ['m', 'q']

for file in file_finder(directory, rho = 0):
    with np.load(file) as data:
        print(file)
        for key in data:
            if key not in excluded:
                pass
                # print(key)
                # print(data[key])


