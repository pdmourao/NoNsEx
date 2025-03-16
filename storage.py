import os
import numpy as np
import json

def npz_file_finder(directory, *args, prints = True, file_spec ='', **kwargs):

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
            with open(json_file_os, mode="r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                for key, value in kwargs_json.items():
                    if data[key] != value:
                        verdict = False
            if verdict:
                file_list.append(npz_file_os)
    if prints:
        print(f'{len(file_list)} file(s) found.')

    return file_list