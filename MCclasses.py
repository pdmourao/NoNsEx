import numpy as np
from time import time
import random
from tqdm import tqdm

# Class HopfieldMC simulates one HopfOfHopfs system
# It holds a state, number of neurons and layers, the patterns and the interaction matrix
# The interaction matrix does not have g plugged in, this is done when actually running the dynamics
# This allows us to initialize the same system for variable lambda outputs

# INPUTS:

# neurons is number of neurons
# L is the number of layers

# K is the list of patterns.
# Dimensions (K, neurons) or:
# If integer K is given, generates K patterns randomly
# If decimal alpha between 0 and 1 is given, generates Floor(alpha * neurons) patterns

# quality is the tuple (ρ, M)
# Constructor calculates r and creates M examples of quality r for each pattern
# SHOULD probably generalize this to include different r's and M's for each layer, if relevant

# sigma is the initial state, dimensions (L, neurons)
# Computes the mixture state of the 3 first patterns if not provided
# Can also receive a tuple of L magnetizations and it will compute a state that has the corresponding diagonal magnetizations

# h is the external field, dimensions (L, neurons)
# It is the scalar that comes multiplied by the external field

# J is the interaction matrix
# If none is provided, it generates examples of the provided quality and computes the supervised interaction matrix
# noise_diff = True gives an independent set of examples for each layer


class HopfieldMC:

    def __init__(self, neurons, K, rho, M, mixM, sigma_type, quality, noise_dif, L = 3, blur = None, h = None,
                 rngSS = np.random.SeedSequence(), compute_J = True, lmb = None):
        t = time()
        self.N = neurons

        self.entropy = rngSS.entropy
        rng = np.random.default_rng(rngSS)

        self.L = L
        # Patterns constructor
        if isinstance(K, int):
            self.pat = rng.choice([-1, 1], (K, self.N))
        else:
            self.pat = K
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
            # Take shape (L, M, K, neurons) for simpler multiplication below
            t0 = time()
            if 'ex' in sigma_type:
                sizeM = max(self.M, mixM)
            else:
                sizeM = self.M
            if noise_dif:
                self.blur = rng.choice([-1, 1], p=[(1 - self.r) / 2, (1 + self.r) / 2],
                                        size=(self.L, sizeM, self.K, self.N))
            else:
                self.blur = np.full(shape = (self.L, sizeM, self.K, self.N), fill_value = rng.choice([-1, 1], p=[(1 - self.r) / 2, (1 + self.r) / 2],
                                             size=(sizeM, self.K, self.N)))
        else:
            self.blur = blur

        self.ex = self.blur[:,:self.M] * self.pat
        self.ex_av = np.average(self.ex, axis=1)

        if compute_J:
            if lmb is None:
                self.J = (1 / (R * self.N)) * np.einsum('kui, luj -> kilj', self.ex_av, self.ex_av)
            else:
                g = np.array([[1, - lmb, - lmb],
                              [- lmb, 1, - lmb],
                              [- lmb, - lmb, 1]])
                self.J = (1 / (R * self.N)) * np.einsum('kl, kui, luj -> kilj', g, self.ex_av, self.ex_av)
            for l in range(self.L):

                for i in range(self.N):
                    self.J[l, i, l, i] = 0
        else:
            self.J = None

        assert sigma_type in ['mix', 'mix_ex', 'dis', 'dis_ex'], 'Non valid sigma_type.'

        if mixM == 0:
            input_ex_av = np.full(shape = (self.L, self.L, self.N), fill_value = self.pat[:self.L])
        else:
            if 'ex' in sigma_type:
                input_blur = self.blur[:, :mixM, :self.L]

            else:
                input_blur = np.full(shape=(self.L, mixM, self.L, self.N),
                                     fill_value=rng.choice([-1, 1], p=[(1 - self.r) / 2, (1 + self.r) / 2],
                                                           size=(mixM, self.L, self.N)))

            input_ex = input_blur * self.pat[:self.L]
            input_ex_av = np.average(input_ex, axis = 1)

        # Initial state
        state = np.zeros(shape=(self.L, self.N))
        state_blur = np.zeros(shape=(len(quality), self.N))

        for idx in range(len(quality)):
            state_blur[idx] = rng.choice([-1, 1], p=[(1 - quality[idx]) / 2, (1 + quality[idx]) / 2], size=self.N)

        if 'dis' in sigma_type:
            for layer in range(self.L):
                state[layer] = np.sign(input_ex_av[layer, layer])
        elif 'mix' in sigma_type:
            state = np.sign(np.sum(input_ex_av, axis = 1))
        self.input = input_ex_av
        self.sigma = state*state_blur

        # External field
        if h is None:
            self.h = np.sign(np.sum(input_ex_av, axis = 1))
        else:
            self.h = h

    # Method simulate runs the MonteCarlo simulation
    # It does L x neurons flips per iteration. Each of these L x neurons flips is one call of the function "dynamics",
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
    # it stops the simulation if less than error * L * neurons spins are flipped in one iteration
    # parallel is optional: if True, it runs parallel dynamics

    # It returns the full history of states
    def simulate(self, beta, H, max_it, error, av_counter, dynamic, J = None, disable = True, prints = False,
                 cut = False, sim_rngSS = None):

        t = time()

        sim_rng = np.random.default_rng(sim_rngSS)

        if J is None:
            J = self.J
        else:
            J = J

        state = self.sigma

        mags = [self.mattis(state)]
        ex_mags = [self.ex_mags(state)]

        saved_idx = 0
        for idx in tqdm(range(max_it), disable = disable):
            saved_idx = idx + 1
            prev_state = state
            state = dynamics(beta = beta, J = J, h = H * self.h, sigma = state, dynamic = dynamic, dyn_rng = sim_rng)
            flips = np.sum(np.abs(state.astype(int) - prev_state.astype(int)))//2
            mags.append(self.mattis(state))
            if prints and disable:
                print(self.mattis(state))
            ex_mags.append(self.ex_mags(state))
            if idx + 2 >= av_counter:
                prev_mags_std = np.std(mags[-av_counter:], axis=0)
                if prints and disable and error >= 1:
                    print(f'{int(flips)} on iteration {idx + 1}.')
                elif prints and disable and error < 1:
                    print(f'Error {np.max(prev_mags_std)} on iteration {idx + 1}')
                if error >= 1 and flips < error:
                    break
                elif np.max(prev_mags_std) < error < 1:
                    break

        if cut:
            return mags[-av_counter:], ex_mags[-av_counter:], saved_idx
        else:
            return mags, ex_mags, saved_idx

    # Method mattis returns an L x L array of the magnetizations with respect to the first L patterns
    def mattis(self, sigma):
        m = (1 / self.N) * np.einsum('li, ui -> lu', sigma, self.pat[:self.L])
        return m

    def ex_mags(self, sigma):
        n = (1 / (self.N*(1+self.rho)*self.r)) * np.einsum('li, lui -> lu', sigma, self.ex_av[:,:self.L])
        return n

# dynamics flips exactly L*neurons neurons
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


def dynamics(beta, J, h, sigma, dynamic = 'sequential', dyn_rng = np.random.default_rng()):

    layers, neurons = np.shape(sigma)
    noise = dyn_rng.uniform(low = -1, high = 1, size = (layers, neurons))

    if dynamic == 'parallel':
        if np.isinf(beta):
            new_sigma = np.sign(np.einsum('kilj,lj->ki', J, sigma) + h)
        else:
            new_sigma = np.sign(np.tanh(beta * (np.einsum('kilj,lj->ki', J, sigma) + h)) + noise)
    elif dynamic == 'sequential':
        new_sigma = sigma.copy()
        neuron_sampling = dyn_rng.permutation(range(neurons))
        for idx_N in neuron_sampling:
            layer_sampling = dyn_rng.permutation(range(layers))
            for idx_L in layer_sampling:
                if np.isinf(beta):
                    new_neuron = np.sign(np.einsum('ki, ki -> ', J[idx_L, idx_N, :, :], new_sigma)
                                        + h[idx_L, idx_N])
                else:
                    new_neuron = np.sign(
                    np.tanh(beta * (np.einsum('ki, ki -> ', J[idx_L,idx_N,:,:], new_sigma)
                                    + h[idx_L, idx_N])) + noise[idx_L, idx_N])
                new_sigma[idx_L, idx_N] = new_neuron
    else:
        raise Exception('No dynamic update rule given.')

    return new_sigma
