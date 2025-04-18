import numpy as np


def Q(beta, lmb, m, xi):
	g = np.array([[   1, -lmb, -lmb],
				  [-lmb,    1, -lmb],
				  [-lmb, -lmb,    1]])
	Qvec = np.zeros((3, 3, 3))
	for mu in range(3):
		for nu in range(3):
			Qvec[:, mu, nu] = xi[mu]*xi[nu]*(np.tanh(beta * g@m@xi))**2
	return Qvec

def EQ(beta, lmb, m):
	resQ = np.zeros((3, 3, 3, 2, 2, 2))
	for idx0, q0 in enumerate([-1, 1]):
		for idx1, q1 in enumerate([-1, 1]):
			for idx2, q2 in enumerate([-1, 1]):
				resQ[:, :, :, idx0, idx1, idx2] = Q(beta, lmb, m, [q0, q1, q2])

	return np.mean(resQ, axis = (3, 4, 5))

eps = 0.1

matm = np.array([[1-eps,   eps,   eps],
			  [  eps, 1-eps,   eps],
			  [  eps,   eps, 1-eps]])
b = 10
l = 0.2

print(EQ(beta = b, lmb = l, m = matm))