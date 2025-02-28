import os
import numpy as np

def file_finder(directory, *args, file_spec = '', **kwargs):

    file_list = []

    for file in os.listdir(directory):
        if 'npz' in file and file_spec in file:
            file_os = os.path.join(directory, file)
            with np.load(file_os) as data:
                verdict = True
                for key, item in kwargs.items():
                    if not np.array_equal(data[key], item):
                        verdict = False
                for idx, item in enumerate(args):
                    if not np.array_equal(data[f'arr_{idx}'], item):
                        verdict = False
                if verdict:
                    file_list.append(file_os)
    print(f'{len(file_list)} file(s) found.')
    return file_list