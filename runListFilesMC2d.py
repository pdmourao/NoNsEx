import numpy as np
from storage import file_finder
from FPfields import m_in

directory = 'MC2d'
excluded = ['H', 'rho']


for file in file_finder(directory, rho = 0):
    with np.load(file) as data:
        print(file)
        print(f'{len(np.load(file[:-1] + 'y'))} sample(s)')
        for key in data:
            if key not in excluded:
                print(key)
                print(data[key])


