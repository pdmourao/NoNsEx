import os
import numpy as np
import json
from functools import reduce
from tqdm import tqdm
import time

def npz_file_finder(directory, *args, prints = False, file_spec ='', **kwargs):

    json_types = [str, bool]
    kwargs_json = {}
    kwargs_num = {}
    file_list = []

    for key, value in kwargs.items():
        if type(value) in json_types:
            kwargs_json[key] = value
        else:
            kwargs_num[key] = value

    full_kwargs_num = dict(kwargs_num)
    for idx, non_kw_arg in enumerate(args):
        full_kwargs_num[f'arr_{idx}'] = non_kw_arg

    for file in os.listdir(directory):
        if 'npz' in file and file_spec in file:
            npz_file_os = os.path.join(directory, file)
            json_file_os = os.path.join(directory, file[:-3] + 'json')

            verdict = True

            with np.load(npz_file_os) as data:
                for key, value in full_kwargs_num.items():
                    if not np.array_equal(data[key], value):
                        verdict = False
            if len(kwargs_json) > 0:
                with open(json_file_os, mode="r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    for key, value in kwargs_json.items():
                        if data[key] != value:
                            verdict = False
            if verdict:
                file_list.append(npz_file_os)
                if prints:
                    print(file)
                    n_samples = len([filenpy for filenpy in os.listdir(directory) if file[:-4] in filenpy and '_m.npy' in filenpy])
                    print(f'Has {n_samples} samples.')
    if prints:
        print(f'{len(file_list)} file(s) found.')

    return file_list

def mathToPython(file, directory = None):
    if directory is None:
        fname = file
    else:
        fname = os.path.join(directory, file)
    with open(fname, 'rb') as f:
        depth = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]
        dims = np.fromfile(f, dtype=np.dtype('int32'), count=depth)
        data = np.transpose(np.reshape(np.fromfile(f, dtype=np.dtype('float64'),
                                                   count=reduce(lambda x, y: x * y, dims)), dims))
    return data

def testfunc(a,b):
    output = (a,b)
    return 10, *output

