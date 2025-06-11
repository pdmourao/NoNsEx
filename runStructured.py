import numpy as np
import matplotlib.pyplot as plt
from MCclasses import HopfieldMC as hop
from time import time

# Carica il file .npz
data = np.load('dataset_cifre.npz') #change with your path

# Estrai immagini e etichette
images = data['images']
labels = data['labels']

images_bin = np.array([-np.sign(vec/255-0.5) for vec in images])

separator = np.all(images_bin[:-1] == images_bin[1:], axis = 0)
picker = np.logical_not(separator)

im_remainder = separator*images_bin[0]
im_reduced = np.array([picker*image for image in images_bin])

indices = [1,2,3]
im_reduced_flat = np.array([im[im!=0] for im in im_reduced])

entropy = 203748477786163793093866656734919284042
rngSS = np.random.SeedSequence(entropy)

t = time()
system = hop(neurons = 5000, K = np.take(im_reduced_flat, indices, axis = 0), rho = 0, M = 1, lmb = 0.3, sigma_type = 'mix',
             noise_dif = False, Jtype = np.float64, prints = True, rngSS = rngSS)
print(f'Generated system in {time() - t} seconds.')
print(f'Entropy: {system.entropy}')
print(f'{system.N} neurons.')
print(f'The intiial state is')
print(f'{system.mattis(system.sigma)}')

states, mags, ex_mags = system.simulate_full(beta = 10, max_it = 100, dynamic = 'sequential', prints = True)