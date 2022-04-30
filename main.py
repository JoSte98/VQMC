"""
@author: Magdalena and Johannes
"""

from VQMC import VQMC
from Optimizer import Optimizer

##### helium
model = VQMC()
optimizer = Optimizer(model)
optimizer.find_optimum()

##### harmonic oscillator
#model = VQMC(num_steps_equilibrate=10000,trial_function=ho_trial, local_energy=ho_local, init_alpha=ho_init_alpha,
#             dimension=ho_dimension, max_step_length=2)
#model.plot_average_local_energies()