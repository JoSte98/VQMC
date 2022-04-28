
from VQMC import VQMC
from harmonic_oscillator import *

##### helium
#model = VQMC(num_steps_equilibrate=20000, num_walkers=100, max_step_length=0.2)
#model.plot_energy()

##### harmonic oscillator
model = VQMC(trial_function=ho_trial, local_energy=ho_local, init_alpha=ho_init_alpha,
             dimension=ho_dimension, max_step_length=0.5)
model.plot_energy()