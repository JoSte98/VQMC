
from VQMC import VQMC
from harmonic_oscillator import *

##### helium
# model = VQMC()
# model.plot_energy()

##### harmonic oscillator
model = VQMC(num_steps_equilibrate=10000, trial_function=ho_trial, local_energy=ho_local, init_alpha=ho_init_alpha, dimension=ho_dimension)
model.plot_energy()