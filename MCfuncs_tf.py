import copy
import tensorflow as tf
from time import time
import random

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



# IsDisentangled checks whether an L x L matrix is disentangled
# This is done by checking every line of the matrix for entries > cutoff
# And then checking if the indices of those entries are all different (thanks Andrea)

def IsDisentangled(m, cutoff):
    m_entries = []
    for m1 in m:
        for idx2, m2 in enumerate(m1):
            if m2 > cutoff:
                m_entries.append(idx2)
    if len(set(m_entries)) == 3:
        return True
    else:
        return False
