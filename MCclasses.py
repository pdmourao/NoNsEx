import numpy as np
from time import time, process_time
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

    def __init__(self, neurons, K, rho, M, sigma_type, noise_dif, lmb = None, quality = [1,1,1], mixM = 0, L = 3, ex = None, h = None,
                 rngSS = np.random.SeedSequence(), compute_J = True, Jtype = np.float64, prints = False):

        self.entropy = rngSS.entropy
        self.rng = np.random.default_rng(rngSS)
        self.L = L
        # Patterns constructor
        if isinstance(K, (int,np.integer)):
            t = time()
            self.pat = self.rng.choice([-1, 1], (max(L,K), neurons))
            if prints:
                print(f'Generated patterns in {time() - t} seconds.')
        else:
            t = time()
            input_K, input_N = np.shape(K)

            if input_N < neurons:
                pat_buffer = self.rng.choice([-1, 1], (input_K, neurons - input_N))
                self.pat = np.concatenate((K, pat_buffer), axis = 1)
                if prints:
                    print(f'Buffered patterns in {time() - t} seconds.')
            else:
                self.pat = K
        # Holds number of patterns

        self.K, self.N = np.shape(self.pat)
        assert self.K >= self.L, 'Should have at least as many patterns as layers.'

        self.rho, self.M = rho, M
        self.r = np.sqrt(1 / (self.rho * self.M + 1))


        # Interaction matrix constructor
        R = self.r**2 + (1 - self.r**2)/self.M

        t = time()
        # axis = 0 performs a sum of matrices, otherwise np.sum starts summing elements
        if ex is None:
            # Examples constructor
            # Define Chi vector
            # Take shape (L, M, K, neurons) for simpler multiplication below
            t = time()
            if 'ex' in sigma_type:
                sizeM = max(self.M, mixM)
            else:
                sizeM = self.M
            if noise_dif:
                blur = self.rng.choice([-1, 1], p=[(1 - self.r) / 2, (1 + self.r) / 2],
                                        size=(self.L, sizeM, self.K, self.N))
            else:
                blur = np.full(shape = (self.L, sizeM, self.K, self.N), fill_value = self.rng.choice([-1, 1], p=[(1 - self.r) / 2, (1 + self.r) / 2],
                                             size=(sizeM, self.K, self.N)))
            if prints:
                print(f'Generated blurs in {time() - t} seconds.')
            self.ex = blur[:, :self.M] * self.pat
            self.ex_av = np.average(self.ex, axis=1).astype(Jtype)
        else:
            self.ex = ex
            self.ex_av = np.average(self.ex, axis=1).astype(Jtype)
            if prints:
                print(f'Computed examples in {time() - t} seconds.')


        if prints:
            print(f'Computed examples and averages in {time() - t} seconds.')

        if compute_J:
            t = time()
            norm = np.float64((1 / (R * self.N))).astype(Jtype)
            if lmb is None:
                self.g = np.ones((3,3))
                self.J = norm * np.einsum('kui, luj -> kilj', self.ex_av, self.ex_av)
            else:
                self.g = np.array([[1, - lmb, - lmb],
                              [- lmb, 1, - lmb],
                              [- lmb, - lmb, 1]])
                self.J = norm * np.einsum('kl, kui, luj -> kilj', self.g.astype(Jtype), self.ex_av, self.ex_av)
            for l in range(self.L):

                for i in range(self.N):
                    self.J[l, i, l, i] = 0
            if prints:
                print(f'Calculated interaction matrix in {time() - t} seconds.')
        else:
            self.J = None

        assert sigma_type in ['mix', 'mix_ex', 'dis', 'dis_ex'], 'Non valid sigma_type.'

        t = time()
        if mixM == 0:
            input_ex_av = np.full(shape = (self.L, self.L, self.N), fill_value = self.pat[:self.L])
        else:
            if 'ex' in sigma_type:
                input_ex = self.ex[:, :mixM, :self.L]

            else:
                input_ex = np.full(shape=(self.L, mixM, self.L, self.N),
                                     fill_value=self.rng.choice([-1, 1], p=[(1 - self.r) / 2, (1 + self.r) / 2],
                                                           size=(mixM, self.L, self.N)))

            input_ex_av = np.average(input_ex, axis = 1)

        # Initial state
        state = np.zeros(shape=(self.L, self.N))
        state_blur = np.zeros(shape=(len(quality), self.N))

        for idx in range(len(quality)):
            state_blur[idx] = self.rng.choice([-1, 1], p=[(1 - quality[idx]) / 2, (1 + quality[idx]) / 2], size=self.N)

        if 'dis' in sigma_type:
            for layer in range(self.L):
                state[layer] = np.sign(input_ex_av[layer, layer])
        elif 'mix' in sigma_type:
            state = np.sign(np.sum(input_ex_av, axis = 1))

        self.input = input_ex_av
        self.sigma = state*state_blur
        if prints:
            print(f'Computed initial state in {time() - t} seconds.')
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

    # It returns the full history of magnetizations
    def simulate(self, beta, max_it, dynamic, H = 0, error = 0, av_counter = 1, J = None, disable = True, prints = False,
                 cut = True, av = False, sim_rngSS = None):

        t = time()
        if sim_rngSS is None:
            sim_rng = np.random.default_rng(np.random.SeedSequence(self.entropy).spawn(1)[0])
        else:
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
            mags = mags[-av_counter:]
            ex_mags = ex_mags[-av_counter:]

        if av:
            mags = np.mean(mags, axis = 0)
            ex_mags = np.mean(ex_mags, axis=0)

        return mags, ex_mags, saved_idx

    def simulate_full(self, beta, max_it, dynamic, H = 0, disable = True, prints = False, sim_rngSS = None):

        t = time()
        if sim_rngSS is None:
            sim_rng = np.random.default_rng(np.random.SeedSequence(self.entropy).spawn(1)[0])
        else:
            sim_rng = np.random.default_rng(sim_rngSS)

        state = self.sigma

        states = [state]
        mags = [self.mattis(state)]
        ex_mags = [self.ex_mags(state)]


        for idx in tqdm(range(max_it), disable = disable):
            prev_state = state
            state = dynamics(beta = beta, J = self.J, h = H * self.h, sigma = state, dynamic = dynamic, dyn_rng = sim_rng)
            flips = np.sum(np.abs(state.astype(int) - prev_state.astype(int)))//2
            mags.append(self.mattis(state))
            if prints and disable:
                print(f'\nIteration {idx+1}:')
                print(self.mattis(state))
            states.append(state)
            ex_mags.append(self.ex_mags(state))

        return states, mags, ex_mags

    # Method mattis returns an L x L array of the magnetizations with respect to the first L patterns
    def mattis(self, sigma):
        m = (1 / self.N) * np.einsum('li, ui -> lu', sigma, self.pat[:self.L])
        return m

    def ex_mags(self, sigma):
        n = (1 / (self.N*(1+self.rho)*self.r)) * np.einsum('li, lui -> lu', sigma, self.ex_av[:,:self.L])
        return n

    def add_load(self, k, prints = False):
        assert self.rho == 0, 'add_load method only written for 0 dataset entropy'
        t = time()
        if prints:
            print(f'System had {self.K} patterns.')
        new_pats = self.rng.choice([-1, 1], (k, self.N))
        self.J = self.J + (1 / self.N) * np.einsum('kl, ui, uj -> kilj', self.g, new_pats, new_pats)
        self.K += np.shape(new_pats)[0]
        t = time() - t
        if prints:
            print(f'Increased to {self.K} in {round(t,2)} seconds.')

# dynamics flips each neuron one time
# Dynamics supported:
# - Non-random sequential (i.e. flips each neuron once, order is random)
# - Parallel
# It is used inside the system classes to run dynamics (method simulate)

# INPUTS
# beta is 1/T
# J is the interaction matrix
# h is the external field
# sigma is the state to be updated
# (optional) parallel = True runs parallel dynamics


def dynamics(beta, J, h, sigma, dynamic, dyn_rng):

    layers, neurons = np.shape(sigma)
    noise = dyn_rng.uniform(low = -1, high = 1, size = (layers, neurons))

    if dynamic == 'parallel':
        if np.isinf(beta):
            return np.sign(np.einsum('kilj,lj->ki', J, sigma) + h)
        else:
            return np.sign(np.tanh(beta * (np.einsum('kilj,lj->ki', J, sigma) + h)) + noise)
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
        return new_sigma
    else:
        raise Exception('No valid dynamic update rule given.')


class TAM:

    def __init__(self, neurons, layers, patterns = 0, rngSS = np.random.SeedSequence(), lmb = None):

        self.fast_noise, self.noise_patterns = tuple(np.random.default_rng(rngSS).spawn(2))

        self._neurons = neurons
        self._layers = layers
        self.patterns = patterns

        self._J = self.interaction(self.patterns)
        self._lmb = lmb

        self.h = np.zeros((layers, neurons))

    @property
    def lmb(self):
        return self._lmb

    @property
    def neurons(self):
        return self._neurons

    @property
    def layers(self):
        return self._layers

    @property
    def J(self):
        return self._J

    @property
    def patterns(self):
        return self._patterns

    @patterns.setter
    def patterns(self, patterns):
        if isinstance(patterns, (int,np.integer)):
            self._patterns = self.noise_patterns.choice([-1, 1], (patterns, self.neurons))
        else:
            self._patterns = patterns

    def interaction(self, data):
        J = (1 / self.neurons) * np.einsum('ki, kj -> ij', data, data)
        if self.lmb is not None:
            J = self.insert_g(J, self.lmb)
        return J

    def mixture(self, n):
        return np.sign(np.sum(self.patterns[:n], axis = 0))

    def insert_g(self, J, lmb):
        assert len(np.shape(J)) == 2, 'Trying to insert lambda into an already full matrix'
        J = np.transpose(np.broadcast_to(J, (self.layers, self.layers, self.neurons, self.neurons))*g(lmb)[:,:,None,None], [0, 2, 1, 3])
        for i in range(self.neurons):
            for l in range(self.layers):
                J[l,i,l,i] = 0
        return J




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

    # It returns the full history of magnetizations
    def simulate(self, beta, max_it, dynamic, H, error, av_counter, g = None, sim_rngSS = None, cut = True, av = True, prints = False):

        t = time()
        if sim_rngSS is None:
            sim_rng = np.random.default_rng(self.fast_noise)
        else:
            sim_rng = np.random.default_rng(sim_rngSS)

        if g is None:
            J = self.J
        else:
            J = np.einsum('kl,ij->klij', g, self.J)

        state = self.sigma

        mags = [self.mattis(state)]
        ex_mags = [self.ex_mags(state)]

        saved_idx = 0
        for idx in range(max_it):
            saved_idx = idx + 1
            prev_state = state
            state = dynamics(beta = beta, J = J, h = H * self.h, sigma = state, dynamic = dynamic, dyn_rng = sim_rng)
            flips = np.sum(np.abs(state.astype(int) - prev_state.astype(int)))//2
            mags.append(self.mattis(state))
            ex_mags.append(self.ex_mags(state))
            if idx + 2 >= av_counter:
                prev_mags_std = np.std(mags[-av_counter:], axis=0)
                if prints and error >= 1:
                    print(f'{int(flips)} on iteration {idx + 1}.')
                elif prints and error < 1:
                    print(f'Error {np.max(prev_mags_std)} on iteration {idx + 1}')
                if error >= 1 and flips < error:
                    break
                elif np.max(prev_mags_std) < error < 1:
                    break

        if cut:
            mags = mags[-av_counter:]
            ex_mags = ex_mags[-av_counter:]

        if av:
            mags = np.mean(mags, axis = 0)
            ex_mags = np.mean(ex_mags, axis=0)

        return mags, ex_mags, saved_idx

    def simulate_full(self, beta, max_it, dynamic, H = 0, disable = True, prints = False, sim_rngSS = None):

        t = time()
        if sim_rngSS is None:
            sim_rng = np.random.default_rng(np.random.SeedSequence(self.entropy).spawn(1)[0])
        else:
            sim_rng = np.random.default_rng(sim_rngSS)

        state = self.sigma

        states = [state]
        mags = [self.mattis(state)]
        ex_mags = [self.ex_mags(state)]


        for idx in tqdm(range(max_it), disable = disable):
            prev_state = state
            state = dynamics(beta = beta, J = self.J, h = H * self.h, sigma = state, dynamic = dynamic, dyn_rng = sim_rng)
            flips = np.sum(np.abs(state.astype(int) - prev_state.astype(int)))//2
            mags.append(self.mattis(state))
            if prints and disable:
                print(f'\nIteration {idx+1}:')
                print(self.mattis(state))
            states.append(state)
            ex_mags.append(self.ex_mags(state))

        return states, mags, ex_mags

    # Method mattis returns an L x L array of the magnetizations with respect to the first L patterns
    def mattis(self, sigma):
        m = (1 / self.neurons) * np.einsum('li, ui -> lu', sigma, self.pat[:self.L])
        return m

    def ex_mags(self, sigma):
        n = (1 / (self.neurons*(1+self.rho)*self.r)) * np.einsum('li, lui -> lu', sigma, self.ex_av[:,:self.L])
        return n

    def add_load(self, k, prints = False):
        assert self.rho == 0, 'add_load method only written for 0 dataset entropy'
        t = time()
        if prints:
            print(f'System had {self.K} patterns.')
        new_pats = self.rng.choice([-1, 1], (k, self.N))
        self.J = self.J + (1 / self.N) * np.einsum('kl, ui, uj -> kilj', self.g, new_pats, new_pats)
        self.K += np.shape(new_pats)[0]
        t = time() - t
        if prints:
            print(f'Increased to {self.K} in {round(t,2)} seconds.')

lmb = np.random.rand()/2
def g(lmb):
    g = np.full((3,3), fill_value = -lmb)
    np.fill_diagonal(g, 1)
    return g