import numpy as np
from FPfields import NoN_q_to_p
from time import time
from tqdm import tqdm

sample = 100000
maxim = 0
print(NoN_q_to_p(b = 10, lmb = 0.1, q = [0.891948, 0.87323223, 0.34945534]) - NoN_q_to_p(b = 10, lmb = 0.1, q = [0.891948, 0.34945534, 0.87323223]))
print(f'Value is {NoN_q_to_p(b = 10, lmb = 0.1, q = [0.891948, 0.87323223, 0.34945534])}')
q_in = np.zeros(shape = 3)
t = np.zeros(shape = sample)
for idx in tqdm(range(sample)):
    q = np.random.uniform(0, 1, size=3)
    t0 = time()
    p = NoN_q_to_p(b = 10, lmb = 0.1, q = q)
    p_rev = NoN_q_to_p(b = 10, lmb = 0.1, q = [q[0], q[2], q[1]])
    t[idx] = time() - t0
    dif = np.abs(p - p_rev)
    if dif > maxim:
        q_in = q
        maxim = dif

print(f'Maximum difference {maxim} (input {q_in}) and average time {np.mean(t)} seconds.')