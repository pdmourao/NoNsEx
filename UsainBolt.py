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
    def __init__(self, func, directory = None, *args, **kwargs):

        # function object ready to compute samples
        # the arguments that will be saved on files
        self._func = func
        self._args = args
        self._kwargs = kwargs

        # other attributes
        self._file_prefix = None
        self._entropy = np.random.SeedSequence().entropy
        self._directory = self.dir(directory)

    @property
    def dir(self):
        return self._directory

    @dir.setter
    def dir(self, directory = None):
        if directory is None:
            self._directory = os.getcwd()
        else:
            input_files = exp_finder(directory=directory, file_spec = self._func.__name__, *self._args, **self._kwargs)
            if len(input_files) > 1:
                print(f'Warning: {len(input_files)} experiment(s) found for given inputs.')
                print(f'{input_files[0]} will be used.')

            # finds existent experiment
            try:
                inputs_file = input_files[0]
                with open(inputs_file[:-3] + 'json', mode="r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    self._entropy = int(data['entropy'])
                print('File found!')
                self._file_prefix = inputs_file[:-10]
            except IndexError:
                print('Experiment not found for given inputs.')

    # call the class to create a new experiment
    def __call__(self):
        if self._file_prefix is None:
            print('Creating new experiment...')
            kwargs_json = {}
            kwargs_num = {}
            for key, value in self._kwargs.items():
                if isjson(value):
                    if isinstance(value, tuple):  # if it's a tuple of strings, pass the list
                        kwargs_json[key] = list(value)
                    else:
                        kwargs_json[key] = value
                else:
                    kwargs_num[key] = value
            print('Starting one.')

            inputs_file = os.path.join(self._directory, f'{self._func.__name__}-{int(time())}_inputs.npz')
            self._file_prefix = inputs_file[:-10]

            with open(f'{inputs_file[:-3]}json', mode="w", encoding="utf-8") as json_file:
                kwargs_json['entropy'] = str(self._entropy)
                json.dump(kwargs_json, json_file)
            np.savez(inputs_file, *self._args, **kwargs_num)
        else:
            print('Experiment already exists.')

    def run(self, sample, save = False, *extra_args, **extra_kwargs):
        output = self._func(entropy = (self._entropy, sample), *self._args, *extra_args, **extra_kwargs, **self._kwargs)
        if save and sample not in self.samples_present:
            np.savez(self._file_prefix+f'sample{sample}', *output)

    def samples_present(self):
        return [int(file.split('_sample')[-1][:-4]) for file in os.listdir(self._directory) if
                                self._file_prefix + 'sample' in os.path.join(self._directory, file)]

    def read_sample(self, sample):
        filename = self._file_prefix + f'sample{sample}.npz'
        with np.load(filename) as data:
            output = tuple(data.values())
        return output

    def read(self):
        output_list = [self.read_sample(sample) for sample in self.samples_present()]
        return map(np.array, zip(*output_list))

    def read_av(self):
        return tuple([np.mean(output, axis =0) for output in self.read()])
