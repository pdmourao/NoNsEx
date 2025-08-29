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
            state = dynamics_old(beta = beta, J = J, h =H * self.h, sigma = state, dynamic = dynamic, dyn_rng = sim_rng)
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
            state = dynamics_old(beta = beta, J = self.J, h =H * self.h, sigma = state, dynamic = dynamic, dyn_rng = sim_rng)
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


def dynamics_old(beta, J, h, sigma, dynamic, dyn_rng):

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

def dynamics(beta, J, h, sigma, dynamic, dyn_rng):

    layers, neurons = np.shape(sigma)
    noise = dyn_rng.uniform(low = -1, high = 1, size = (layers, neurons))

    if dynamic == 'parallel':
        if np.isinf(beta):
            return np.sign(np.einsum('klij,lj->ki', J, sigma) + h)
        else:
            return np.sign(np.tanh(beta * (np.einsum('klij,lj->ki', J, sigma) + h)) + noise)
    elif dynamic == 'sequential':
        new_sigma = sigma.copy()
        neuron_sampling = dyn_rng.permutation(range(neurons))
        for idx_N in neuron_sampling:
            layer_sampling = dyn_rng.permutation(range(layers))
            for idx_L in layer_sampling:
                if np.isinf(beta):
                    new_neuron = np.sign(np.einsum('ki, ki -> ', J[idx_L, :, idx_N, :], new_sigma)
                                        + h[idx_L, idx_N])
                else:
                    new_neuron = np.sign(
                    np.tanh(beta * (np.einsum('ki, ki -> ', J[idx_L,:,idx_N,:], new_sigma)
                                    + h[idx_L, idx_N])) + noise[idx_L, idx_N])
                new_sigma[idx_L, idx_N] = new_neuron
        return new_sigma
    else:
        raise Exception('No valid dynamic update rule given.')

def g(layers, lmb):
    matrix_g = np.full((layers, layers), -lmb)
    np.fill_diagonal(matrix_g, 1)
    return matrix_g


class TAM:

    def __init__(self, neurons, layers, split, supervised, r, m, k = 0, patterns = None, lmb = -1, rng_ss = np.random.SeedSequence()):

        # usage of SeedSequence objects allows for reproducibility
        # create one seed sequence for each independent source of randomness
        rng_seeds = rng_ss.spawn(3)

        # fast noise uses a seed sequence, since simulate always starts from the initial state
        # in order to get independent runs of the same system, one should further spawn independent seeds from this
        # simulate should do this by default, but one should keep it in mind nonetheless
        self.fast_noise = rng_seeds[0]
        # slow noise already uses bit generators since the method add_patterns always starts from where it left off
        # if we used a seed sequence, it would start from the beginning everytime we called methods like add_pattern
        # for reproducibility, one should always keep in mind the different seeds and where each one "starts"
        self.noise_patterns = np.random.default_rng(rng_seeds[1])
        self.noise_examples = np.random.default_rng(rng_seeds[2])

        self._neurons = neurons
        self._layers = layers

        self._r = r
        self._m = m

        # we set k at 0 first because the pattern setter can only be called for k = 0
        self._k = 0

        # initializes the patterns (see patterns setter)
        # it is used when we want to give specific patterns and not randomly generate them
        # for instance when copying them from another system
        # after this we are only allowed to add more patterns by increasing k and they are randomly generated
        # examples for each pattern are randomly generated (see patterns setter)
        if patterns is not None:
            self.patterns = patterns
        else:
            self._k = 0
            self._patterns = self.gen_patterns(0)

        # available examples
        self._examples = self.gen_examples(self._patterns)

        # variables for the computation of the interaction matrix
        self._lmb = lmb
        self._split = split
        self._supervised = supervised

        # interaction matrix and effective examples
        # effective examples are the examples above but split among layers (if split) and average among examples (if supervised)
        # gets defined when interaction matrix does (see set_interaction)
        self._J = None
        self._effective_examples = None


        # next we add random patterns and examples
        # can only be higher than existing patterns
        if k > self._k:
            self.add_patterns(k - self.k) # this one already constructs interaction matrix by itself
        else:
            # interaction matrix gets constructed with the set_interaction method
            self.set_interaction()
        if k < self._k:
            print('Invalid k input will be ignored.')

        # the initial state and external field to be used in simulate
        # defined manually outside the constructor
        # can use the methods mix, dis, etc (define them as needed)
        self.external_field = None
        self.initial_state = None

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
    def r(self):
        return self._r

    @property
    def m(self):
        return self._m

    @property
    def J(self):
        return self._J

    @property
    def patterns(self):
        return self._patterns

    @property
    def examples(self):
        return self._examples

    @property
    def k(self):
        return self._k

    @property
    def m_per_layer(self):
        if self._split:
            if self._m % self._layers != 0:
                print('Warning: non-integer m per layer.')
            return self.m // self._layers
        else:
            return self._m


    @patterns.setter
    def patterns(self, patterns):
        assert self._k == 0, 'Patterns have already been set. Use add_patterns instead.'
        assert isinstance(patterns, np.ndarray) and len(patterns.shape) == 2 and patterns.shape[1] == self._neurons, 'Invalid pattern input.'
        self._k = patterns.shape[0]
        self._patterns = patterns
        self._examples = self.gen_examples(self._patterns)

    def add_patterns(self, k):
        assert isinstance(k, int) and k > 0, 'Number of patterns can only be increased.'
        extra_patterns = self.gen_patterns(k)
        extra_examples = self.gen_examples(extra_patterns)
        self._patterns = np.concatenate((self._patterns, extra_patterns))
        self._examples = np.concatenate((self._examples, extra_examples), axis=1)

        if self._J is not None:
            self._J = self._J + self.interaction(extra_examples)
            self._effective_examples = self.effective_examples()
        else:
            self.set_interaction()
        self._k += k

    # the initial state setter will allow us to choose initial states by their names
    def state(self, name = None, **kwargs):
        if name is None:
            return np.zeros((self._layers, self._neurons))
        else:
            return name(**kwargs)

    # constructor of the interaction matrix (also used when patterns are added)
    # for cases where all layers are the same (ie rho = 0 or not-split), and lambda is not given, this is a neurons * neurons matrix
    # this way, one can save memory and reuse the same system for different lambda's
    # otherwise it has dimensions layers * layers * neurons * neurons with lambda = -1
    # simulate allows for a matrix J input because of this
    # and the method insert_g allows us to insert the lambda dependence latter

    # the reason this method and set_interaction are not the same method is for this one to be used for the extra patterns in add_patterns
    def interaction(self, examples = None):
        if examples is None:
            examples = self._examples
        big_r = self._r ** 2 + (1 - self._r ** 2) / self.m_per_layer
        if self._lmb >= 0 or self._split: # in these cases the interaction matrix already has full dimensions
            eff_examples = self.effective_examples(examples)
            if self._supervised:
                J = (1 / (big_r * self.neurons)) * np.einsum('kl, kui, luj -> klij', self.g(self._lmb), eff_examples, eff_examples)
            else:
                J = (1 / (big_r * self.neurons * self.m_per_layer)) * np.einsum('kl, kaui, lauj -> klij', self.g(self._lmb), eff_examples,
                                                           eff_examples)
            for i in range(self.neurons):
                for l in range(self.layers):
                    J[l, l, i, i] = 0
        else: # in these cases we keep the interaction matrix with only dimensions neurons * neurons and add the g matrix later
            if self._supervised:
                av_examples = np.mean(examples, axis = 0)
                J = (1 / (big_r * self.neurons)) * np.einsum('ui, uj -> ij', av_examples, av_examples)
            else:
                J = (1 / (big_r * self.neurons * self._m)) * np.einsum('aui, auj -> ij', examples, examples)
        return J

    def effective_examples(self, examples = None):
        if examples is None:
            examples = self._examples
        k = np.shape(examples)[1]
        if self._split:
            applied_examples = np.reshape(examples, (self._layers, self.m_per_layer, k, self._neurons))
        else:
            applied_examples = np.broadcast_to(examples, (self._layers, self._m, k, self._neurons))
        if self._supervised:
            applied_examples = np.mean(applied_examples, axis=1)
        return applied_examples


    def set_interaction(self, lmb = None, split = None, supervised = None):
        if lmb is not None:
            self._lmb = lmb
        if split is not None:
            self._split = split
        if supervised is not None:
            self._supervised = supervised
        self._J = self.interaction()
        self._effective_examples = self.effective_examples()


    def gen_patterns(self, k):
        return self.noise_patterns.choice([-1, 1], (k, self._neurons))

    def gen_examples(self, patterns):
        k = np.shape(patterns)[0]
        blurs = self.noise_examples.choice([-1, 1], p=[(1 - self._r) / 2, (1 + self._r) / 2], size = (self._m, k, self._neurons))
        return blurs * patterns

    # mixture state
    def mix(self, n = None):
        if n == 0:
            return np.zeros((self._layers, self._neurons))
        if n is None:
            n = self._layers
        return np.broadcast_to(np.sign(np.sum(self._patterns[:n], axis = 0)), (self._layers, self._neurons))

    # disentangled state
    def dis(self):
        return self._patterns[:self._layers]

    def g(self, lmb):
        return g(layers = self._layers, lmb = lmb)

    def insert_g(self, lmb):
        if len(np.shape(self._J)) == 2:
            J = np.broadcast_to(self._J, (self.layers, self.layers, self.neurons, self.neurons))*self.g(lmb)[:,:,None,None]
            for i in range(self.neurons):
                for l in range(self.layers):
                    J[l, l, i, i] = 0
        else:
            J = self._J*self.g(lmb)[:, :, None, None]
        return J


    # Method mattis returns an L x L array of the magnetizations with respect to the first L patterns
    def mattis(self, sigma, cap = None):
        if cap is None:
            cap = self._layers
        return (1 / self._neurons) * np.einsum('li, ui -> lu', sigma, self._patterns[:cap])

    def ex_mags(self, sigma, cap = None):
        if cap is None:
            cap = self._layers
        if self._supervised:
            big_r = self._r ** 2 + (1 - self._r ** 2) / self.m_per_layer
            return (self._r / (self._neurons * big_r)) * np.einsum('li, lui -> lu', sigma, self._effective_examples[:,:cap])
        else:
            # is there a constant here?
            return (1 / self._neurons) * np.einsum('li, alui -> alu', sigma, self._effective_examples[:,:,cap])

    # Method simulate runs the MonteCarlo simulation
    # It does L x neurons flips per iteration.
    # Each of these L x neurons flips is one call of the function "dynamics" (defined above)
    # At each iteration it appends the new state a list
    # It loops until a maximum number of iterations is reached
    # Or until the standard deviation in the last av_counter magnetizations is below a certain threshold

    # INPUTS:
    # max_it is the maximum number of iterations
    # beta is the inverse temperature
    # dynamic is either 'parallel' or 'sequential' (see function dynamics)
    # H is the strength of the external field (the external field already exists in self.h)
    # lmb is the value of lambda, in case the interaction matrix does not have it yet
    # error is the threshold for the standard deviation to assert convergence
    # av_counter is the  number of iterations used in the standard deviation / convergence test
    # av = True takes the average of the last av_counter iterations before returning, otherwise it returns the full history

    # It returns the full history of magnetizations
    def simulate(self, beta, max_it, dynamic, error, av_counter, h_norm, sim_J = None, av = True, sim_rng = None):
        assert self.initial_state is not None, 'Initial state not provided.'
        assert self.external_field is not None, 'External field not provided.'

        if av_counter == 1 and error > 0:
            print('Warning: av_counter set to 1 with positive error')

        if sim_rng is None:
            dyn_rng = np.random.default_rng(self.fast_noise.spawn(1)[0])
        else:
            dyn_rng = np.random.default_rng(sim_rng)

        if sim_J is None:
            assert self._lmb >= 0, r'\lambda not available to simulate.'
            sim_J = self._J

        state = self.initial_state

        mags = [self.mattis(state)]
        ex_mags = [self.ex_mags(state)]

        idx = 0
        while idx < max_it: # do the simulation
            idx += 1
            state = dynamics(beta = beta, J = sim_J, h = h_norm * self.external_field, sigma = state, dynamic = dynamic, dyn_rng = dyn_rng)
            mags.append(self.mattis(state))
            ex_mags.append(self.ex_mags(state))
            if idx + 1 >= av_counter: # size of the actual arrays has +1 since they include initial states
                prev_mags_std = np.std(mags[-av_counter:], axis=0)
                if np.max(prev_mags_std) < error:
                    break

        if av:
            mags = np.mean(mags[-av_counter:], axis = 0)
            ex_mags = np.mean(ex_mags[-av_counter:], axis=0)

        return mags, ex_mags, idx


# test if adding patterns / examples is actually worth it compared to restarting
# see if saved samples are retrievable with this program
# use UsainBolt