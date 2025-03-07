import numpy as np
from storage import file_finder

directory = 'MC2d_old'
excluded = ['lmb']

for file in file_finder(directory, rho = 0.2):
    with np.load(file) as data:
        print(file)
        for key in data:
            if key not in excluded:
                print(key)
                print(data[key])
        try:
            mattis_trials = np.load(file[:-1] + 'y')
            print(f'{len(mattis_trials)} samples')
        except FileNotFoundError:
            pass


