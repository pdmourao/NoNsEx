import tensorflow as tf
from time import time
import tensorflow_probability as tfp

tfd = tfp.distributions

# Class HopfOfHopfsMC simulates one HopfOfHopfs system
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

class HopfieldMC_tf:

    def __init__(self, N, L, pat, quality, dtype = tf.float32, h = None, blur = None, sigma = None, noise_dif = False):

        b0 = tfd.Sample(tfd.Bernoulli(probs = 1/2))

        self.N = N

        self.dtype = dtype

        self.L = L
        # Patterns constructor
        if isinstance(pat, int):
            self.pat = b0.sample([pat, self.N])
        else:
            self.pat = pat
        # Holds number of patterns
        self.K = tf.shape(self.pat)[0]
        assert self.K >= self.L, 'Should have at least as many patterns as layers.'

        self.rho, self.M = quality
        self.r = tf.math.sqrt(1 / (self.rho * self.M + 1))

        # Interaction matrix constructor
        R = self.r ** 2 + (1 - self.r ** 2) / self.M

        # axis = 0 performs a sum of matrices, otherwise np.sum starts summing elements
        if blur is None:
            br = tfd.Sample(tfd.Bernoulli(probs = (1+self.r)/2))
            # Examples constructor
            # Define Chi vector
            # Take shape (L, M, K, N) for simpler multiplication below
            t0 = time()
            if noise_diff:
                self.blur = br.sample(shape = [self.L, self.M, self.K, self.N])
            else:
                self.blur = tf.experimental.numpy.full(shape=[self.L, self.M, self.K, self.N],
                                                       fill_value= br.sample([self.M, self.K, self.N]))
        else:
            self.blur = blur
        self.ex = tf.cast(self.blur * self.pat, dtype = self.dtype)
        ex_av = tf.math.reduce_mean(self.ex, axis=1)

        diagonal_killer = tf.tensordot(tf.linalg.diag(tf.zeros([self.L]), padding_value = 1),
                                       tf.linalg.diag(tf.zeros([self.N]), padding_value = 1),
                                       axes = 0)

        self.J = (1 / (R * self.N)) * tf.einsum('kui, luj, klij -> kilj', ex_av, ex_av,
                                                tf.cast(diagonal_killer, dtype = ex_av.dtype))

        # Initial state
        if sigma is None:
            self.sigma = tf.experimental.numpy.full([self.L, self.N], fill_value = tf.math.sign(tf.math.reduce_sum(self.pat[:self.L], axis=0)))

        else:
            self.sigma = sigma

        # External field
        if h is None:
            self.h = tf.cast(tf.experimental.numpy.full(shape=[self.L, N], fill_value = tf.math.sign(tf.math.reduce_sum(self.pat[:self.L], axis=0))), self.dtype)
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
    # it stops the simulation if less then error * L * N spins are flipped in one iteration
    # parallel is optional: if True, it runs parallel dynamics

    # It returns the full history of states
    def simulate(self, max_it, T, H, J, av_counter = 5, parallel = True, error = None):
        J_tf = tf.convert_to_tensor(J, dtype = self.dtype)
        t = time()
        # J_tf = tf.convert_to_tensor(J, tf.float32)
        # h_tf = tf.convert_to_tensor(self.h, tf.float32)
        # print(f'Convertions took {time() - t} seconds.')
        sigma_h = [self.sigma]
        prev_mags = [self.mattis(self.sigma)]
        for i in range(max_it):

            new_sigma_out= dynamics(beta = 1/T, J = J, h = H * self.h, sigma = sigma_h[-1], dtype = self.dtype)

            sigma_h.append(new_sigma_out)
            prev_mags.append(self.mattis(new_sigma_out))
            if i > av_counter - 3:
                prev_mags_var = tf.math.reduce_variance(tf.convert_to_tensor(prev_mags[-av_counter:]), axis=0)
                if error is not None and tf.math.reduce_max(prev_mags_var) < error:
                    break

        prev_mags_av = tf.math.reduce_mean(tf.convert_to_tensor(prev_mags[-av_counter:]), axis=0)

        return sigma_h, prev_mags_av

    # Method mattis returns an L x L array of the magnetizations with respect to the first L patterns
    def mattis(self, sigma):
        m = (1 / self.N) * tf.einsum('li, ui -> lu', tf.cast(sigma, self.dtype), tf.cast(self.pat[:self.L], self.dtype))
        return m


def dynamics(beta, J, h, sigma, dtype):

    new_sigma = tf.Variable(sigma)
    layers, neurons = sigma.get_shape()
    noise = tf.cast(tf.random.uniform(minval = -1, maxval = 1, shape = [layers, neurons]), dtype)

    new_sigma = tf.sign(tf.tanh(beta * (tf.einsum('kilj,lj->ki', J, tf.cast(new_sigma, dtype)) +
                                        h)) + noise)

    return new_sigma
