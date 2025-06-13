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

im_reduced_flat = np.array([im[im!=0] for im in im_reduced])
print(np.shape(im_reduced_flat))

fig, axes = plt.subplots(2,5)
for idx, ax in enumerate(np.ndarray.flatten(axes)):
    ax.imshow(im_reduced[idx], vmin=-1, vmax=1)
    ax.set_title(f'{labels[idx]}')
    ax.set_axis_off()
plt.show()

fig, ax = plt.subplots(1)
ax.imshow(im_remainder, vmin=-1, vmax=1)
ax.set_title(f'Remainder')
ax.set_axis_off()
plt.show()