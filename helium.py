import numpy as np

def helium_trial(position_1, position_2, alpha):
    r12 = np.abs(position_1-position_2)
    return np.exp(-2*np.abs(position_1))*np.exp(-2*np.abs(position_2))*np.exp(r12/(2*(1 + alpha*r12)))

def helium_local(position_1, position_2, alpha):
    r_hat_1 = position_1/np.abs(position_1)
    r_hat_2 = position_1/np.abs(position_2)
    r12 = np.abs(position_1 - position_2)
    return -4 + (r_hat_1 - r_hat_2) * (position_1 - position_2) * (1/(r12*(1 + alpha*r12)**2)) \
            - 1/(r12*(1 + alpha*r12)**3) - 1/(4*(1 + alpha*r12)**4) + 1/r12

helium_init_alpha = 10

helium_dimension = 6