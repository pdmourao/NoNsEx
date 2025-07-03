import os
from time import time
from storage import exp_finder, isjson
import json
import numpy as np

# This class is supposed to handle running general experiments, saving them to files, etc
# And also facilitate parallel processing across samples
# So it's a runner (as in it runs simulations) and supposed to be fast, so, naturally, it's called UsainBolt.

class Experiment:

    # f is the experiment function to be done
    # exception are kwargs keys that are not parameters meant to be saved as inputs
    # handle is the string to be used in the file names, f's name by default
    # spanned_vars are the variable parameters across each sample
    # args and kwargs are arguments of f
    def __init__(self, func, directory, exceptions, handle = None, create = False, *args, **kwargs):

        # function object ready to compute samples
        self.func = lambda entropy: func(entropy, *args, **kwargs)
        self.directory = directory

        # name of the files either starts with func.__name__-#Number or with handle-#Number
        if handle is None:
            handle = func.__name__

        input_files = exp_finder(directory=directory, file_spec=handle, *args, **kwargs)
        if len(input_files) > 1:
            print(f'Warning: {len(input_files)} experiment(s) found for given inputs.')
            print(f'{input_files[0]} will be used.')

        try:
            inputs_file = input_files[0]
            with open(inputs_file[:-3] + 'json', mode="r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                entropy_from_os = int(data['entropy'])
            print('File found!')
            self.file_prefix = inputs_file[:-10]
            self.entropy = entropy_from_os

        except IndexError:
            print('No experiments found for given inputs.')
            entropy_from_os = np.random.SeedSequence().entropy
            self.entropy = entropy_from_os
            self.file_prefix = None
            if create:
                # separate inputs into those destined for json files and those destined for npz files
                kwargs_json = {}
                kwargs_num = {}
                for key, value in kwargs.items():
                    if isjson(value):
                        if isinstance(value, tuple): # if it's a tuple of strings, pass the list
                            kwargs_json[key] = list(value)
                        else:
                            kwargs_json[key] = value
                    else:
                        kwargs_num[key] = value

                print('Starting one.')

                inputs_file = os.path.join(directory, f'{handle}-{int(time())}_inputs.npz')
                self.file_prefix = inputs_file[:-10]

                with open(f'{inputs_file[:-3]}json', mode="w", encoding="utf-8") as json_file:
                    kwargs_json['entropy'] = str(entropy_from_os)
                    json.dump(kwargs_json, json_file)
                np.savez(inputs_file, *args, **kwargs_num)

    def run(self, sample, save = False):
        output = self.func((self.entropy, sample))
        if save and sample not in self.samples_present:
            if isinstance(output, tuple):
                np.savez(self.file_prefix+f'sample{sample}', *output)
            else:
                np.save(self.file_prefix+f'sample{sample}', output)

    def samples_present(self):
        return [int(file.split('_sample')[-1][:-4]) for file in os.listdir(self.directory) if
                                self.file_prefix + f'sample' in os.path.join(self.directory, file)]

    def read(self):
        # code reader method


