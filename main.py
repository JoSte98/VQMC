"""
@author: Magdalena and Johannes
"""

from VQMC import VQMC
from Optimizer import Optimizer

##### helium
model = VQMC()
model.alpha_energy_dependence(stop=0.25, steps=9, start=0.05)
optimizer = Optimizer(model)
optimizer.find_optimum()

##### harmonic oscillator
#model = VQMC(model='LHO',init_alpha=0.1)
#model.get_alpha_energy_dependence(0.6,10)
#model.get_energy_mean()
#model.plot_average_local_energies()
#print(model.expected_energy)