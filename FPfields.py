import numpy as np
from scipy import special, integrate
from time import time, perf_counter
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

# functions to be used as inputs for the fixed point method
# some of them are used just as auxiliary for the main ones that actually serve as inputs
# It is easy to distinguish them: the latter ones should always return a dictionary
# This way the FixedPoint.iterate method can feed the output as input directly
# And it works cleanly regardless of how many outputs / inputs are necessary

# @timeit
def gauss_int(f, errorbound):
    lowerbound = special.erfinv(-1 + errorbound)
    upperbound = special.erfinv(1 - errorbound)

    return integrate.quad(lambda theta: (1 / np.sqrt(2 * np.pi)) * np.exp(-theta ** 2 / 2) * f(theta),
                          lowerbound, upperbound)[0]

def partmHopEx(n, beta, rho, errorbound):
    lowerbound = special.erfinv(-1 + errorbound)
    upperbound = special.erfinv(1 - errorbound)

    partm = (1/4) * integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2) * (
        np.tanh(beta*(n[0]+n[1]+n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n))
        +np.tanh(beta*(n[0]+n[1]-n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n))
        +np.tanh(beta*(n[0]-n[1]+n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n))
        +np.tanh(beta*(n[0]-n[1]-n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n))), lowerbound, upperbound)[0]
    return partm

def partqHopEx(n, beta, rho, errorbound = 0):
    lowerbound = special.erfinv(-1 + errorbound)
    upperbound = special.erfinv(1 - errorbound)

    partq = (1/4) * integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2) * (
        (np.tanh(beta*(n[0]+n[1]+n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n)))**2
        +(np.tanh(beta*(n[0]+n[1]-n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n)))**2
        +(np.tanh(beta*(n[0]-n[1]+n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n)))**2
        +(np.tanh(beta*(n[0]-n[1]-n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n)))**2), lowerbound, upperbound)[0]
    return partq


def HopEx(m, q, n = None, *, beta, rho, errorbound = 0, alpha=0, H=0):

    if n is None:
        n = m / (1 + rho - rho * beta * (1 - q))

    new_m = np.array([partmHopEx(n, beta = beta, rho = rho, errorbound = errorbound),
                      partmHopEx([n[1], n[0], n[2]],beta = beta, rho = rho, errorbound = errorbound),
                      partmHopEx([n[2], n[1], n[0]], beta = beta, rho = rho, errorbound = errorbound)])
    new_q = partqHopEx(n, beta = beta, rho = rho, errorbound = errorbound)
    new_n = (new_m/(1+rho))+beta*(rho/(1+rho))*(1-new_q)*n

    return new_m, new_q, new_n


def CW(m, T, h):
    m_out = np.tanh((m+h)/T)
    return m_out


def integrandHLm(x, m, q, p, alpha, beta, h):
    return np.tanh(beta*(m+h+x*(np.sqrt(alpha * p))))


def integrandHLq(x, m, q, p, alpha, beta, h):
    return (np.tanh(beta*(m+h+x*(np.sqrt(alpha * p)))))**2


def integrandHLq_simp(x, q, alpha, beta):
    return (np.tanh(beta*x*(np.sqrt(alpha * q / ((1 - beta * (1 - q)) ** 2)))))**2


def HLH(m, q, p, alpha, beta, h, errorbound = 0):
    new_m = gauss_int(lambda theta: integrandHLm(theta, m, q, p, alpha, beta, h), errorbound)
    if new_m < 1e-6:
        new_m=0
    new_q = gauss_int(lambda theta: integrandHLq(theta, m, q, p, alpha, beta, h), errorbound)
    new_p = q / ((1 - beta * (1 - q)) ** 2)
    return new_m, new_q, new_p

def HLH_onlyq(m, q, alpha, beta, errorbound = 0):
    new_q = gauss_int(lambda theta: integrandHLq_simp(theta, q, alpha, beta), errorbound)
    return 0, new_q


def NoNsLL(m, lmb, T, H):
    m1 = NoNsLL1(m[0], m[1], lmb, 1/T, H)
    m2 = NoNsLL2(m[0], m[1], lmb, 1/T, H)
    return m1, m2


def NoNsLL1(m1, m2, lmb, beta, H):
    Tpp = np.tanh(beta*((m1+2*m2)*(1-2*lmb)+H))
    Tpm = np.tanh(beta*(m1-2*lmb*m2+H))
    Tmm = np.tanh(beta*(m1-2*m2+2*lmb*m1-H))
    return (1/4)*(Tpp+2*Tpm+Tmm)


def NoNsLL2(m1, m2, lmb, beta, H):
    Tpp = np.tanh(beta*((m1+2*m2)*(1-2*lmb)+H))
    Tpm = np.tanh(beta*(m1-lmb*2*m2+H))
    Tmp = np.tanh(beta * (2*m2-m1 - lmb*2*m1 + H))
    Tmm = np.tanh(beta*(-m1+2*lmb*m2-H))
    result = (1/4)*(Tpp+Tpm+Tmp+Tmm)
    return result


def NoNsLLdiag(m, lmb, T, H):
    beta = 1/T
    Tpp = np.tanh(beta*(m*(1-2*lmb)+H))
    Tpm = np.tanh(beta*(m+H))
    Tmp = np.tanh(beta * (m+H))
    Tmm = np.tanh(beta*(m*(1+2*lmb)-H))
    return (1/4)*(Tpp+Tpm+Tmp+Tmm)


def g(lmb):
    return np.array([[1, -lmb, -lmb],
                     [-lmb, 1, -lmb],
                     [-lmb, -lmb, 1]])


def ginv(lmb):
    return np.linalg.inv(g(lmb))


def NoNsM(x1, x2, h, errorbound = 0):
    lowerbound = special.erfinv(-1 + errorbound)
    upperbound = special.erfinv(1 - errorbound)

    Tpp = integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                                   * np.tanh(x1[0] + x1[1] + x1[2] + x2 * z + h), lowerbound, upperbound)[0]
    Tpm = integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                                   * np.tanh(x1[0] + x1[1] - x1[2] + x2 * z + h), lowerbound, upperbound)[0]
    Tmp = integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                                   * np.tanh(x1[0] - x1[1] + x1[2] + x2 * z + h), lowerbound, upperbound)[0]
    Tmm = integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                                   * np.tanh(x1[0] - x1[1] - x1[2] + x2 * z -h), lowerbound, upperbound)[0]

    return np.mean([Tpp, Tpm, Tmp, Tmm])

def NoNsMall(x1, x2, h, errorbound=0):
    return np.array([NoNsM(x1, x2, h, errorbound),
                     NoNsM([x1[1], x1[0], x1[2]], x2, h, errorbound),
                     NoNsM([x1[2], x1[1], x1[0]], x2, h, errorbound)])


def NoNsQ(x1, x2, h, errorbound = 0):
    lowerbound = special.erfinv(-1 + errorbound)
    upperbound = special.erfinv(1 - errorbound)

    Tpp2 = integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                                    * (np.tanh(x1[0] + x1[1] + x1[2] + x2 * z +h) ** 2), lowerbound, upperbound)[0]
    Tpm2 = integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                                    * (np.tanh(x1[0] + x1[1] - x1[2] + x2 * z +h) ** 2), lowerbound, upperbound)[0]
    Tmp2 = integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                                    * (np.tanh(x1[0] - x1[1] + x1[2] + x2 * z +h) ** 2), lowerbound, upperbound)[0]
    Tmm2 = integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                                    * (np.tanh(x1[0] - x1[1] - x1[2] + x2 * z -h) ** 2), lowerbound, upperbound)[0]
    return np.mean([Tpp2, Tpm2, Tmp2, Tmm2])


# Lines are patterns, columns are layers
def NoNsEx(m, q, n, beta, lmb, rho, alpha, H, errorbound = 0, p_reg = 1):

    gn = g(lmb) @ n

    sd_rho = np.sqrt(rho) * np.linalg.norm(gn, axis = 1)
    m_out = np.zeros(shape = (3, 3))
    q_out = np.zeros(shape=3)
    for layer in range(3):
        m_out[layer] = NoNsMall(x1 = beta * gn[layer], x2 = beta * sd_rho[layer], h = beta * H, errorbound = errorbound)
        q_out[layer] = NoNsQ(x1 = beta * gn[layer], x2 = beta * sd_rho[layer], h = beta * H, errorbound = errorbound)

    # This is the same as the loop above
    # m_out = vNoNsM(x1 = beta * gn, x2 = beta * sd_rho, h = beta * H, errorbound = errorbound)
    # q_out = vNoNsQ(x1=beta * gn, x2=beta * sd_rho, h=beta * H, errorbound=errorbound)
    n_out = (1/(1+rho))*(m_out + beta * rho * np.diag(1-q_out) @ gn)

    return m_out, q_out, n_out

def m_in(epsilon=0):
    return np.full(shape = (3, 3), fill_value = 1/2) - np.full(shape = (3, 3), fill_value = epsilon) + np.diag(np.full(3, fill_value = 2*epsilon))

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


v = np.array([1,-1])
ev_vec = cartesian_product(v,v,v)

def bin_sum(f):
    return np.sum(np.apply_along_axis(f, axis = 1, arr = ev_vec))

def bin_ev(f):
    return (1/8)*np.sum(np.apply_along_axis(f, axis = 1, arr = ev_vec))

def ZL_fac(alpha, beta, g_matrix, m, C_no_ab, p12, xi, s, theta, h):
    return np.exp((1 / 2) * np.dot(s, alpha*beta*C_no_ab @ s) + beta * np.dot((g_matrix@m@xi + h) + theta * np.sqrt(alpha * p12.diagonal()), s))

def ZL_weight(alpha, beta, g_matrix, m, C_no_ab, p12, xi, s, theta, h):
    return ZL_fac(alpha, beta, g_matrix, m, C_no_ab, p12, xi, s, theta, h)/bin_sum(lambda s_den: ZL_fac(alpha, beta, g_matrix, m, C_no_ab, p12, xi, s_den, theta, h))

def NoNsEx_HL_m(m, C, p12, alpha, beta, g_matrix, h, errorbound = 0):
    new_m = np.zeros((3,3))
    for layer in range(3):
        for pat in range(3):
            new_m[layer,pat] = bin_ev(lambda xi:
                                      bin_sum(lambda s:
                                              gauss_int(lambda theta:
                                                        xi[pat] * s[layer] *
                                                        ZL_weight(alpha, beta, g_matrix, m, C, p12, xi, s, theta, h)
                                                        ,errorbound = errorbound)))
    return new_m


def NoNsEx_HL_q(m, C, p12, alpha, beta, g_matrix, h, errorbound = 0):
    new_q = np.ones((3,3))
    single_s = np.array([[bin_ev(lambda xi:
                                 gauss_int(lambda theta:
                                           bin_sum(lambda s:
                                        s[layer1] * ZL_weight(alpha, beta, g_matrix, m, C, p12, xi, s, theta, h))
                                *bin_sum(lambda s:
                                        s[layer2] * ZL_weight(alpha, beta, g_matrix, m, C, p12, xi, s, theta, h)
                                        ), errorbound = errorbound
                                ))
                          for layer1 in range(3)] for layer2 in range(3)])

    for l1 in range(3):
        for l2 in range(3):
            if l1 != l2:
                new_q[l1,l2] = bin_ev(lambda xi:
                                      bin_sum(lambda s:
                                              gauss_int(lambda theta:
                                                        s[l1] * s[l2] *
                                                            ZL_weight(alpha, beta, g_matrix, m, C, p12, xi, s, theta, h)
                                                            ,errorbound = errorbound)))

    mat_for_diag = np.array([[np.sqrt(p12[k,k]/p12[l,l])*(new_q[l,k] - single_s[l,k])
                              for k in range(3)]
                             for l in range(3)])

    np.fill_diagonal(new_q, 1 - np.sum(mat_for_diag, axis = 1))

    return new_q


def NoNsEx_HL(m, q, p, alpha, beta, lmb, h, errorbound = 0):

    Ct = np.diag(beta * (1 - np.diag(q)))
    g_matrix = g(lmb)
    Ginv = np.linalg.inv(np.linalg.inv(g_matrix)-Ct)

    new_m = NoNsEx_HL_m(m, Ginv, p, alpha, beta, g_matrix, h, errorbound = errorbound)
    new_q = NoNsEx_HL_q(m, Ginv, p, alpha, beta, g_matrix, h, errorbound=errorbound)
    new_p = Ginv @ q @ Ginv

    return new_m, new_q, new_p

def input_checker(m, q, p, alpha, beta, lmb, h, errorbound = 0):
    print(alpha, beta, lmb, h)

pert_spur = np.array([[ 1, -1, -1],
                      [ 1, -1, -1],
                      [ 1, -1, -1]])
pert_dis = np.array([[ 1, -1, -1],
                     [-1,  1, -1],
                     [-1,  1, -1]])
initial_q_LL = np.full(shape = 3, fill_value = 1.)
initial_q_i = np.full(shape = (3,3), fill_value = 1.)
initial_q_o = np.diag(initial_q_LL)
initial_p_i = np.full(shape = (3,3), fill_value = 1.)
initial_p_o = np.diag(initial_q_LL)

print(integrate.quad(lambda theta: (1 / np.sqrt(2 * np.pi)) * np.exp(-theta ** 2 / 2) * np.tanh(10*(-0.2 + theta*np.sqrt(0.16*1))), -np.inf, np.inf)[0])

