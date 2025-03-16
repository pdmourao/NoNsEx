from npy_append_array import NpyAppendArray
import numpy as np

arr1 = np.array([[[1, 2], [3, 4]], [[3, 4], [5, 6]]])
arr2 = np.array([[[7, 8], [9, 19]], [[34, 45], [54, 61]]])

filename = 'out.npy'

with NpyAppendArray(filename, delete_if_exists=True) as npaa:
	npaa.append(arr1)
	npaa.append(arr2)
	npaa.append(arr2)

data = np.load(filename, mmap_mode="r")

print(data)
print(np.shape(data))