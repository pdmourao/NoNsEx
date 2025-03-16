from storage import npz_file_finder
import FPfuncs as fp
from FPfields import m_in, initial_q
import numpy as np
from matplotlib import pyplot as plt


rho_values = np.linspace(0, 0.3, 100, endpoint = False)
lmb_values = np.linspace(0, 0.5, 100, endpoint = False)

beta = 10
cutoff = 0.9
pert_4 = np.array([[ 1, -1, -1],
                   [-1,  1, -1],
                   [-1, -1,  1]])

pert = 1e-8*pert_4



tr_dets = [lambda m: fp.disentangle_det(m, cutoff)]
# tr_dets = []
tr_arrays_rho = [[] for det in tr_dets]
tr_arrays_lmb = [[] for det in tr_dets]

initial_m = m_in()

for file in npz_file_finder('FP1d', file_spec='NoNsEx_', arr_0 =m_in() + pert, arr_1 = initial_q, H = 0,
							alpha = 0, rho = rho_values, beta = 10):
    with np.load(file) as data:
        for idx_det, tr_det in enumerate(tr_dets):
            idx_tr = fp.FindTransition(data['m'], tr_det=tr_det)
            if idx_tr > 0:
                tr_arrays_rho[idx_det].append([data['rho'][idx_tr], data['lmb']])

for file in npz_file_finder('FP1d', file_spec='NoNsEx_', arr_0 =m_in() + pert, arr_1 = initial_q, H = 0,
							alpha = 0, lmb = lmb_values, beta = 10):
    with np.load(file) as data:
        for idx_det, tr_det in enumerate(tr_dets):
            idx_tr = fp.FindTransition(data['m'], tr_det=tr_det)
            if idx_tr > 0:
                tr_arrays_lmb[idx_det].append([data['rho'], data['lmb'][idx_tr]])

for tr_array in tr_arrays_rho:
    trans_sorted = sorted(tr_array, key=lambda x: x[0])
    final_trans = list(map(list, zip(*trans_sorted)))
    rho_array, lmb_array = final_trans[0], final_trans[1]
    plt.scatter(rho_array, lmb_array, color = 'black')

for tr_array in tr_arrays_lmb:
    trans_sorted = sorted(tr_array, key=lambda x: x[0])
    final_trans = list(map(list, zip(*trans_sorted)))
    rho_array, lmb_array = final_trans[0], final_trans[1]
    plt.scatter(rho_array, lmb_array, color = 'black')

plt.xlabel('$\\rho$')
plt.ylabel('$\lambda$')
plt.title(f'$\\beta$ = {beta} mixture state transitions')

#        if interpolate_bool:
#            interpolator = make_interp_spline(l_array, b_array)
#            l_values_smooth = np.linspace(start=l_array[0], stop=l_array[-1], num=500, endpoint=True)
#            plt.plot(l_values_smooth, interpolator(l_values_smooth))
#        else:
#            plt.scatter(l_array, b_array)
plt.xlim(0, 0.3)
plt.ylim(0, 0.5)
plt.show()