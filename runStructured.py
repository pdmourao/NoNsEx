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

# fig, axes = plt.subplots(2,5)
# for idx, ax in enumerate(np.ndarray.flatten(axes)):
#     ax.imshow(images_bin[idx], vmin=0, vmax=1)
#     ax.set_title(f'{labels[idx]}')
#     ax.set_axis_off()
# plt.show()

patterns = np.array([np.reshape(vec, newshape = (-1)) for vec in images_bin])

t = time()
system = hop(neurons = np.size(images[0]), K = patterns, rho = 15, M = 50, lmb = 0.1, sigma_type = 'mix',
             noise_dif = True, Jtype = np.float16, prints = True)
print(f'Generated system in {time() - t} seconds.')

states, mags, ex_mags = system.simulate_full(beta = np.inf, max_it = 100, dynamic = 'sequential', prints = True)