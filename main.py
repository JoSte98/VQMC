
from VQMC import VQMC
from harmonic_oscillator import *

##### helium
model = VQMC(num_steps_equilibrate=30000, num_walkers=400, max_step_length=0.5)
model.plot_energy()

##### harmonic oscillator
#model = VQMC(num_steps_equilibrate=10000,trial_function=ho_trial, local_energy=ho_local, init_alpha=ho_init_alpha,
#             dimension=ho_dimension, max_step_length=2)
model.plot_energy()