"""
@author: Magdalena and Johannes
"""

from VQMC import VQMC
from Optimizer import Optimizer

##### helium
#model = VQMC()
#model.alpha_energy_dependence(stop=0.25, steps=20, start=0.05)
#optimizer = Optimizer(model)
#optimizer.find_optimum()

##### harmonic oscillator
#model = VQMC(model='LHO', init_alpha=0.3)
#model.alpha_energy_dependence(stop=0.7, steps=20, start=0.3)
#optimizer = Optimizer(model, gradient_method="2nd derivative")
#optimizer.find_optimum()

##### hydrogen
model = VQMC(model='Hydrogen', init_alpha=0.8)
model.alpha_energy_dependence(stop=1.2, steps=20, start=0.8)
#optimizer = Optimizer(model, gradient_method="2nd derivative")
#optimizer.find_optimum()