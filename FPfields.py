import numpy as np
import scipy
from time import time
import math

# functions to be used as inputs for the fixed point method
# some of them are used just as auxiliary for the main ones that actually serve as inputs
# It is easy to distinguish them: the latter ones should always return a dictionary
# This way the FixedPoint.iterate method can feed the output as input directly
# And it works cleanly regardless of how many outputs / inputs are necessary

def partmHopEx(n, beta, rho, errorbound):
    lowerbound = scipy.special.erfinv(-1 + errorbound)
    upperbound = scipy.special.erfinv(1 - errorbound)

    partm = (1/4)*scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2) * (
        np.tanh(beta*(n[0]+n[1]+n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n))
        +np.tanh(beta*(n[0]+n[1]-n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n))
        +np.tanh(beta*(n[0]-n[1]+n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n))
        +np.tanh(beta*(n[0]-n[1]-n[2])+z*beta*np.sqrt(rho)*np.linalg.norm(n))), lowerbound, upperbound)[0]
    return partm

def partqHopEx(n, beta, rho, errorbound = 0):
    lowerbound = scipy.special.erfinv(-1 + errorbound)
    upperbound = scipy.special.erfinv(1 - errorbound)

    partq = (1/4)*scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2) * (
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


def integrandHLm(x, m, q, h, alfa, beta):
    return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)*np.tanh(beta*(m+h+x*(np.sqrt(alfa*q)/(1-beta*(1-q)))))


def integrandHLq(x, m, q, h, alfa, beta):
    return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)*(np.tanh(beta*(m+h+x*(np.sqrt(alfa*q)/(1-beta*(1-q))))))**2


def HLH(m, alfa, T, h):
    q1 = scipy.integrate.quad(integrandHLm, -np.inf, np.inf, args = (m[0], m[1], h, alfa, 1/T))[0]
    q2 = scipy.integrate.quad(integrandHLq, -np.inf, np.inf, args = (m[0], m[1], h, alfa, 1/T))[0]
    return q1, q1


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


def T_LL(a, lmb, T, H, m):  # gives the np array [Tpp, Tpm, Tmp, Tmm]
    beta = 1/T
    Tlist = np.empty(4)
    Tlist[0] = np.tanh(beta * (np.dot(g(lmb)[a], m[0]+m[1]+m[2]) + H))
    Tlist[1] = np.tanh(beta * (np.dot(g(lmb)[a], m[0]+m[1]-m[2]) + H))
    Tlist[2] = np.tanh(beta * (np.dot(g(lmb)[a], m[0]-m[1]+m[2]) + H))
    Tlist[3] = np.tanh(beta * (np.dot(g(lmb)[a], m[0]-m[1]-m[2]) - H))
    return Tlist


def NoNsLLFull(m, q, lmb, T, H):
    new_m = np.empty((3, 3))
    new_q = np.zeros(3)
    for a in range(3):
        new_m[0][a] = np.mean(T_LL(a, lmb, T, H, m))
        new_m[1][a] = np.mean(T_LL(a, lmb, T, H, [m[1], m[0], m[2]]))
        new_m[2][a] = np.mean(T_LL(a, lmb, T, H, [m[2], m[1], m[0]]))
    return new_m, new_q


def NoNsM(x1, x2, h, errorbound = 0):
    lowerbound = scipy.special.erfinv(-1 + errorbound)
    upperbound = scipy.special.erfinv(1 - errorbound)

    Tpp = scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                     * np.tanh(x1[0] + x1[1] + x1[2] + x2 * z + h), lowerbound, upperbound)[0]
    Tpm = scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                     * np.tanh(x1[0] + x1[1] - x1[2] + x2 * z + h), lowerbound, upperbound)[0]
    Tmp = scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                     * np.tanh(x1[0] - x1[1] + x1[2] + x2 * z + h), lowerbound, upperbound)[0]
    Tmm = scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                     * np.tanh(x1[0] - x1[1] - x1[2] + x2 * z -h), lowerbound, upperbound)[0]

    return np.mean([Tpp, Tpm, Tmp, Tmm])

def NoNsMall(x1, x2, h, errorbound=0):
    return np.array([NoNsM(x1, x2, h, errorbound),
                     NoNsM([x1[1], x1[0], x1[2]], x2, h, errorbound),
                     NoNsM([x1[2], x1[1], x1[0]], x2, h, errorbound)])

vNoNsM = np.vectorize(NoNsMall, signature='(n),(),(),()->(n)')

def NoNsQ(x1, x2, h, errorbound = 0):
    lowerbound = scipy.special.erfinv(-1+errorbound)
    upperbound = scipy.special.erfinv(1-errorbound)

    Tpp2 = scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                     * (np.tanh(x1[0] + x1[1] + x1[2] + x2 * z +h) ** 2), lowerbound, upperbound)[0]
    Tpm2 = scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                     * (np.tanh(x1[0] + x1[1] - x1[2] + x2 * z +h) ** 2), lowerbound, upperbound)[0]
    Tmp2 = scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                     * (np.tanh(x1[0] - x1[1] + x1[2] + x2 * z +h) ** 2), lowerbound, upperbound)[0]
    Tmm2 = scipy.integrate.quad(lambda z: (1 / np.sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2)
                     * (np.tanh(x1[0] - x1[1] - x1[2] + x2 * z -h) ** 2), lowerbound, upperbound)[0]
    return np.mean([Tpp2, Tpm2, Tmp2, Tmm2])

vNoNsQ = np.vectorize(NoNsQ, signature = '(n),(),(),()->()')

# Lines are patterns, columns are layers
def NoNsEx(m, q, n = None, *, beta, lmb, rho, alpha, H, errorbound = 0, p_reg = 1):

    if n is None:
        n = np.linalg.inv(NoNsEx_NtoM(q, beta, rho, lmb)) @ m


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

def NoNsEx_NtoM(q, beta, rho, lmb):
    return (1+rho)*(np.eye(3) - beta*(rho/(1+rho))*np.diag(1-q) @ g(lmb))

vNoNsEx_NtoM = np.vectorize(NoNsEx_NtoM, signature = '(d,d),(d),(),()->(d,d)')


def NoNsExOld(m, q, T, lmb, rho, alpha, H, p_reg = 1):
    t = time()
    matrix_g = g(lmb)
    beta = 1/T
    # print(np.eye(3) - (beta * rho / (1 + rho)) * matrix_g @ np.diag(1-q))
    n = (1/(1+rho)) * m @ np.linalg.inv(np.eye(3)-(beta * rho / (1 + rho)) * matrix_g @ np.diag(1-q))
    input_gn = n @ matrix_g
    input_ab_p = np.zeros(3)
    # input_ab_p = alpha * beta * NoN_q_to_p(beta, lmb, q, p_reg)
    # print(input_ab_p)
    input_q = np.sqrt(rho * (np.linalg.norm(input_gn, axis = 0)**2) + input_ab_p)
    print(input_q)
    m_out = np.empty((3, 3))
    q_out = np.empty(3)
    # print(NoNsM(beta * np.array([input_gn[0, 0], input_gn[1, 0], input_gn[2, 0]]), input_q[0], beta * H))
    for l in range(3):
        m_out[0,l] = NoNsM(beta * np.array([input_gn[0,l], input_gn[1,l], input_gn[2,l]]), beta * input_q[l], beta * H)
        m_out[1, l] = NoNsM(beta * np.array([input_gn[1,l], input_gn[0,l], input_gn[2,l]]), beta * input_q[l], beta * H)
        m_out[2, l] = NoNsM(beta * np.array([input_gn[2,l], input_gn[1,l], input_gn[0,l]]), beta * input_q[l], beta * H)
        q_out[l] = NoNsQ(beta * np.array([input_gn[0,l], input_gn[1,l], input_gn[2,l]]), beta * input_q[l], beta * H)
    return m_out, q_out

def NoNsEx_noalpha(m, q, T, lmb, rho, alpha, H, p_reg = 1):
    t = time()
    matrix_g = g(lmb)
    beta = 1/T
    n = (1/(1+rho)) * m @ np.linalg.inv(np.eye(3)-(beta * rho / (1 + rho)) * matrix_g @ np.diag(1-q))
    input_gn = n @ matrix_g
    input_ab_p = np.zeros(3)
    # print(input_ab_p)
    input_q = np.sqrt(rho * (np.linalg.norm(input_gn, axis = 0)**2) + input_ab_p)
    # print(input_q)
    m_out = np.empty((3, 3))
    q_out = np.empty(3)
    # print(NoNsM(beta * np.array([input_gn[0, l], input_gn[1, l], input_gn[2, l]]), input_q[l], beta * H))
    for l in range(3):
        m_out[0,l] = NoNsM(beta * np.array([input_gn[0,l], input_gn[1,l], input_gn[2,l]]), input_q[l], beta * H)
        m_out[1, l] = NoNsM(beta * np.array([input_gn[1,l], input_gn[0,l], input_gn[2,l]]), input_q[l], beta * H)
        m_out[2, l] = NoNsM(beta * np.array([input_gn[2,l], input_gn[1,l], input_gn[0,l]]), input_q[l], beta * H)
        q_out[l] = NoNsQ(beta * np.array([input_gn[0,l], input_gn[1,l], input_gn[2,l]]), input_q[l], beta * H)
    return m_out, q_out


def NoN_q_to_p1(b, lmb, q, p_reg):
    q0, q1, q2 = tuple(q)
    result = -(b*lmb**2*q1*(1 - 2*lmb)**2*(b*(lmb + 1)*(q2 - 1) + 1)**2*(2*lmb**2 + lmb - 1)**3 + b*lmb**2*q2*(1 - 2*lmb)**2*(b*(lmb + 1)*(q1 - 1) + 1)**2*(2*lmb**2 + lmb - 1)**3 + 2*b*lmb**2*np.sqrt(q1*q2)*(1 - 2*lmb)**2*(b*(lmb + 1)*(q1 - 1) + 1)*(b*(lmb + 1)*(q2 - 1) + 1)*(2*lmb**2 + lmb - 1)**3 - 2*b*lmb*np.sqrt(q0*q1)*(2*lmb - 1)*(lmb**2 - (-b*(q1 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1)*(-b*(q2 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1))*(b*(lmb + 1)*(q2 - 1) + 1)*(2*lmb**2 + lmb - 1)**3 - 2*b*lmb*np.sqrt(q0*q2)*(2*lmb - 1)*(lmb**2 - (-b*(q1 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1)*(-b*(q2 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1))*(b*(lmb + 1)*(q1 - 1) + 1)*(2*lmb**2 + lmb - 1)**3 + b*q0*(lmb**2 - (-b*(q1 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1)*(-b*(q2 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1))**2*(2*lmb**2 + lmb - 1)**3 + lmb*np.sqrt(q1/q0)*(lmb**2*(2*lmb - 1)*(b*(lmb + 1)*(q0 - 1) + 1) + lmb**2*(2*lmb - 1)*(b*(lmb + 1)*(q1 - 1) + 1) - (lmb**2 - (-b*(q0 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1)*(-b*(q1 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1))*(-b*(q2 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1))**2 + lmb*np.sqrt(q2/q0)*(lmb**2*(2*lmb - 1)*(b*(lmb + 1)*(q0 - 1) + 1) + lmb**2*(2*lmb - 1)*(b*(lmb + 1)*(q1 - 1) + 1) - (lmb**2 - (-b*(q0 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1)*(-b*(q1 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1))*(-b*(q2 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1))**2)/((-2*lmb**2 - lmb + 1)*(lmb**2*(2*lmb - 1)*(b*(lmb + 1)*(q0 - 1) + 1) + lmb**2*(2*lmb - 1)*(b*(lmb + 1)*(q1 - 1) + 1) - (lmb**2 - (-b*(q0 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1)*(-b*(q1 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1))*(-b*(q2 - 1)*(2*lmb**2 + lmb - 1) - lmb + 1))**2)
    if math.isnan(result):
        return 1
    else:
        return result

def NoN_q_to_p(b, lmb, q, p_reg = 1):
    p0 = NoN_q_to_p1(b, lmb, q, p_reg)
    p1 = NoN_q_to_p1(b, lmb, [q[2], q[0],q[1]], p_reg)
    p2 = NoN_q_to_p1(b, lmb, [q[1], q[2], q[0]], p_reg)
    return np.array([p0, p1, p2])

def m_in(epsilon=0):
    return np.full(shape = (3, 3), fill_value = 1/2) - np.full(shape = (3, 3), fill_value = epsilon) + np.diag(np.full(3, fill_value = 2*epsilon))

initial_q = np.full(shape = 3, fill_value = 1)

