"""
@author: Magdalena and Johannes
"""

from VQMC import VQMC

##### helium
model = VQMC()
E = model.get_energy_mean_value(4000)
model.plot_average_local_energies()
print(E)

##### harmonic oscillator
#model = VQMC(num_steps_equilibrate=10000,trial_function=ho_trial, local_energy=ho_local, init_alpha=ho_init_alpha,
#             dimension=ho_dimension, max_step_length=2)
#model.plot_average_local_energies()