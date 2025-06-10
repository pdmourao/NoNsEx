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

def isolator(image_array):
    n_images, size_x, size_y = np.shape(image_array)
    images_reduced = np.zeros_like(image_array)
    image_remainder = np.zeros_like(image_array[0])
    counter_same = 0
    counter_dif = 0
    for pixel_x in range(size_x):
        for pixel_y in range(size_y):
            if all([image[pixel_x,pixel_y] == image_array[0, pixel_x,pixel_y] for image in image_array]):
                image_remainder[pixel_x, pixel_y] = image_array[0, pixel_x,pixel_y]
                counter_same += 1
            else:
                for image_idx in range(n_images):
                    images_reduced[image_idx, pixel_x, pixel_y] = image_array[image_idx, pixel_x, pixel_y]
                counter_dif += 1
    print(counter_same)
    print(counter_dif)
    return images_reduced, image_remainder

# define flatten reduced image
# define image filler

im_reduced, im_remainder = isolator(images_bin)
# print(im_reduced)
# print(im_remainder)

fig, axes = plt.subplots(2,5)
for idx, ax in enumerate(np.ndarray.flatten(axes)):
    ax.imshow(im_reduced[idx], vmin=0, vmax=1)
    ax.set_title(f'{labels[idx]}')
    ax.set_axis_off()
plt.show()

fig, ax = plt.subplots(1)
ax.imshow(im_remainder, vmin=0, vmax=1)
ax.set_title(f'Remainder')
ax.set_axis_off()
plt.show()