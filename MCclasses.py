import numpy as np
from time import time
import random
from tqdm import tqdm

# Class HopfieldMC simulates one HopfOfHopfs system
# It holds a state, number of neurons and layers, the patterns and the interaction matrix
# The interaction matrix does not have g plugged in, this is done when actually running the dynamics
# This allows us to initialize the same system for variable lambda outputs

# INPUTS:

# N is number of neurons
# L is the number of layers

# pat is the list of patterns.
# Dimensions (K, N) or:
# If integer K is given, generates K patterns randomly
# If decimal alpha between 0 and 1 is given, generates Floor(alpha * N) patterns

# quality is the tuple (ρ, M)
# Constructor calculates r and creates M examples of quality r for each pattern
# SHOULD probably generalize this to include different r's and M's for each layer, if relevant

# sigma is the initial state, dimensions (L, N)
# Computes the mixture state of the 3 first patterns if not provided
# Can also receive a tuple of L magnetizations and it will compute a state that has the corresponding diagonal magnetizations

# h is the external field, dimensions (L, N)
# It is the scalar that comes multiplied by the external field

# J is the interaction matrix
# If none is provided, it generates examples of the provided quality and computes the supervised interaction matrix
# noise_diff = True gives an independent set of examples for each layer


class HopfieldMC:

    def __init__(self, N, L, pat, rho, M, lmb = None, blur = None, h = None, sigma = None, noise_dif = False):
        t = time()
        self.N = N

        self.L = L
        # Patterns constructor
        if isinstance(pat, int):
            self.pat = np.random.choice([-1, 1], (pat, self.N))
        else:
            self.pat = pat
        # Holds number of patterns
        self.K = np.shape(self.pat)[0]
        assert self.K >= self.L, 'Should have at least as many patterns as layers.'

        self.rho, self.M = rho, M
        self.r = np.sqrt(1 / (self.rho * self.M + 1))


        # Interaction matrix constructor
        R = self.r**2 + (1 - self.r**2)/self.M

        # axis = 0 performs a sum of matrices, otherwise np.sum starts summing elements
        if blur is None:
            # Examples constructor
            # Define Chi vector
            # Take shape (L, M, K, N) for simpler multiplication below
            t0 = time()
            if noise_dif:
                self.blur = np.random.choice([-1, 1], p=[(1 - self.r) / 2, (1 + self.r) / 2],
                                        size=(self.L, self.M, self.K, self.N))
            else:
                self.blur = np.full(shape = (self.L, self.M, self.K, self.N), fill_value = np.random.choice([-1, 1], p=[(1 - self.r) / 2, (1 + self.r) / 2],
                                             size=(self.M, self.K, self.N)))
        else:
            self.blur = blur

        self.ex = self.blur * self.pat
        ex_av = np.average(self.ex, axis=1)

        if lmb is None:
            self.J = (1 / (R * self.N)) * np.einsum('kui, luj -> kilj', ex_av, ex_av)
        else:
            g = np.array([[1, - lmb, - lmb],
                          [- lmb, 1, - lmb],
                          [- lmb, - lmb, 1]])
            self.J = (1 / (R * self.N)) * np.einsum('kl, kui, luj -> kilj', g, ex_av, ex_av)

        for l in range(self.L):

            for i in range(self.N):
                self.J[l, i, l, i] = 0

            # Initial state
            if sigma is None:
                self.sigma = np.full((self.L, self.N), np.sign(np.sum(self.pat[:self.L], axis=0)))
            elif isinstance(sigma, tuple):
                self.sigma = np.zeros(shape=(self.L, self.N))
                assert len(sigma) == self.L, 'm input does not match number of layers'
                for idx in range(self.L):
                    self.sigma[idx] = self.pat[idx] * np.random.choice([-1, 1],
                                                                       p=[(1 - sigma[idx]) / 2, (1 + sigma[idx]) / 2],
                                                                       size=self.N)
            else:
                self.sigma = sigma

        # External field
        if h is None:
            self.h = np.full(shape = (self.L, N), fill_value = np.sign(np.sum(self.pat[0:self.L], axis = 0)))
        else:
            self.h = h

    # Method simulate runs the MonteCarlo simulation
    # It does L x N flips per iteration. Each of these L x N flips is one call of the function "dynamics",
    # defined in the file MCfuncs.py (See below)
    # At each iteration it appends the new state to the list sigma_h
    # It loops until a maximum number of iterations is reached
    # or until the number of spins that are flipped is lower than a given threshold
    # (this is what serves as a convergence test)

    # INPUTS:
    # max_it is the maximum number of iterations
    # T is the temperature
    # H is the strength of the external field (the external field already exists in self.h)
    # J is the interaction matrix
    # Despite self.J existing, it is given here as an input to allow the matrix g to be plugged in
    # error is optional
    # it stops the simulation if less than error * L * N spins are flipped in one iteration
    # parallel is optional: if True, it runs parallel dynamics

    # It returns the full history of states
    def simulate(self, beta, H, max_it, error, av_counter, parallel, J = None, disable = True, cut = False):
        t = time()

        if J is None:
            J = self.J
        else:
            J = J
        state = self.sigma

        mags = [self.mattis(state)]

        for idx in tqdm(range(max_it), disable = disable):

            state = dynamics(beta = beta, J = J, h = H * self.h, sigma = state, parallel = parallel)

            mags.append(self.mattis(state))

            if idx + 2 >= av_counter:
                prev_mags_std = np.std(mags[-av_counter:], axis = 0)
                if np.max(prev_mags_std) < error:
                    break
        if cut:
            return mags[-av_counter:]
        else:
            return mags

    # Method mattis returns an L x L array of the magnetizations with respect to the first L patterns
    def mattis(self, sigma):
        m = (1 / self.N) * np.einsum('li, ui -> lu', sigma, self.pat[:self.L])
        return m

    def ex_mags(self, sigma):
        ex_av = np.average(self.ex, axis=1)
        n = (1 / (self.N*(1+self.rho)*self.r)) * np.einsum('li, lui -> lu', sigma, ex_av[:,:self.L])
        return n

# dynamics flips exactly L*N neurons
# Dynamics supported:
# - Non-random sequential (i.e. flips each neuron once)
# - Parallel
# It is used inside the system classes to run dynamics (method simulate)

# INPUTS
# beta is 1/T
# J is the interaction matrix
# h is the external field
# sigma is the state to be updated
# (optional) parallel = True runs parallel dynamics

def dynamics(beta, J, h, sigma, parallel = False):

    layers, neurons = np.shape(sigma)
    noise = np.random.uniform(low = -1, high = 1, size = (layers, neurons))


    if parallel:
        new_sigma = np.sign(np.tanh(beta * (np.einsum('kilj,lj->ki', J, sigma) + h)) + noise)
    else:
        new_sigma = sigma.copy()
        neuron_sampling = random.sample(range(neurons), neurons)
        for idx_N in neuron_sampling:
            layer_sampling = random.sample(range(layers), layers)
            for idx_L in layer_sampling:
                new_neuron = np.sign(
                    np.tanh(beta * (np.einsum('ki, ki -> ', J[idx_L,idx_N,:,:], new_sigma)
                                    + h[idx_L, idx_N])) + noise[idx_L, idx_N])
                new_sigma[idx_L, idx_N] = new_neuron

    return new_sigma


class HopfieldMC_rho:

    def __init__(self, N, L, pat, rho, M, blur = None, sigma = None):
        self.N = N

        if isinstance(pat, int):
            self.pat = np.random.choice([-1, 1], size = (pat, N))
            self.K = pat
        else:
            self.pat = pat
            self.K = len(pat)

        self.rho, self.M = rho, M

        self.r = 1/np.sqrt(1+self.rho * self.M)

        if blur is None:
            self.blur = np.random.choice([-1, 1], p = [(1-self.r)/2, (1+self.r)/2], size = (self.M, self.K, N))
        else:
            self.blur = blur

        self.ex = self.blur*self.pat
        avex = np.average(self.ex, axis = 0)
        R = self.r**2 + ((1-self.r**2)/self.M)

        if sigma is None:
            self.sigma = np.sign(np.sum(self.pat[:L], axis=0))
        else:
            self.sigma = sigma

        self.J = (1/(N * R)) * np.einsum('ki,kj->ij', avex, avex)
        for i in range(self.N):
            self.J[i, i] = 0
        if sigma is None:
            self.sigma = np.sign(np.sum(self.pat[:L], axis = 0))
        else:
            self.sigma = sigma

        J_terms = (1 / (N * R)) * np.einsum('ki,kj->kij', avex, avex)
        for k in range(self.K):
            for i in range(self.N):
                pass
                # J_terms[k, i, i] = 0

        print(R)
        ex_norm = np.array([(1/N)*np.dot(avex[u], avex[u]) for u in range(self.K)])
        print(ex_norm)

        for term in [1, 2]:
            print(f'For term {term}')
            print('The term for it is')
            print(self.ex_mags(J_terms[term]@self.sigma)[term])
            print('The rest is')
            print(sum([self.ex_mags(J_terms[other]@self.sigma)[term] for other in range(self.K) if other != term]))




    def mattis(self, sigma):
        # print(np.shape(self.pat))
        # print(np.shape(sigma))
        return (1/self.N)*(self.pat @ sigma)

    def ex_mags(self, sigma):
        ex_av = np.average(self.ex, axis=0)
        n = (1 / (self.N*(1+self.rho)*self.r)) * (ex_av @ sigma)
        return n

    def simulate(self, beta, max_it, error = 0, parallel = False, disable = True):
        sigma_h = [self.sigma]
        for _ in tqdm(range(max_it), disable = disable):
            noise = np.random.uniform(low = -1, high = 1, size = self.N)
            # print('next')
            # self.beta = 1000
            # print(self.mattis(np.sign(np.tanh(self.beta * (self.J @ new_sigma)))))
            if parallel:
                new_sigma = np.sign(np.tanh(beta * (self.J @ sigma_h[-1])) + noise)
                print('Field ex mags')
                field = np.sign((self.J @ sigma_h[-1]))
                print(self.ex_mags(field))
            else:
                new_sigma = sigma_h[-1].copy()
                neuron_sampling = random.sample(range(self.N), self.N)
                for idx_N in neuron_sampling:
                    new_neuron = np.sign(np.tanh(beta*(np.dot(self.J[idx_N], new_sigma)))+noise[idx_N])
                    new_sigma[idx_N] = new_neuron

            sigma_h.append(new_sigma)
        return sigma_h

