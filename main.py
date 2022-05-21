"""
Main file including plots of the potential and the minimization for all 3 examples.

@author: Magdalena and Johannes
"""

from VQMC import VQMC
from Optimizer import Optimizer

def main():
    """

    :return: 0 if successful
    """
    ####################################################################################################################
    ##### HELIUM ATOM
    #### initialize the model
    model_He = VQMC(max_step_length=0.3, Focker_Planck=True)
    
    #### optimize and find the minimal energy
    optimizer_He = Optimizer(model_He, gradient_method="2nd derivative")
    optimizer_He.find_optimum()
    print("The minimum energy: ", optimizer_He.min_E)
    print("The error of the minimum energy: ", optimizer_He.min_variance)
    print("The minimum alpha: ", optimizer_He.min_alpha)
    
    #### get and plot the potential (alpha-energy dependence)
    model_He.alpha_energy_dependence(stop=0.25, steps=20, start=0.05)
    
    
    ####################################################################################################################
    ##### 1D LINEAR HARMONIC OSCILLATOR
    #### initialize the model
    model_LHO = VQMC(model='LHO', init_alpha=0.175, max_step_length=1.0, num_walkers=400, Focker_Planck=True)
    
    #### optimize and find the minimal energy
    optimizer_LHO = Optimizer(model_LHO, gradient_method="2nd derivative")
    optimizer_LHO.find_optimum()
    print("The minimum energy: ",optimizer_LHO.min_E)
    print("The error of the minimum energy: ",optimizer_LHO.min_variance)
    print("The minimum alpha: ",optimizer_LHO.min_alpha)
    
    #### get and plot the potential (alpha-energy dependence)
    model_LHO.alpha_energy_dependence(stop=0.7, steps=20, start=0.3)
    
    
    ####################################################################################################################
    ##### HYDROGEN ATOM
    #### initialize the model
    model_H = VQMC(model='Hydrogen', init_alpha=0.8, Focker_Planck=True)
    
    #### optimize and find the minimal energy
    optimizer_H = Optimizer(model_H, gradient_method="2nd derivative")
    optimizer_H.find_optimum()
    print("The minimum energy: ",optimizer_H.min_E)
    print("The error of the minimum energy: ",optimizer_H.min_variance)
    print("The minimum alpha: ",optimizer_H.min_alpha)
    
    #### get and plot the potential (alpha-energy dependence)
    model_H.alpha_energy_dependence(stop=1.2, steps=20, start=0.8)

    return 0

if __name__ == "__main__":
    main()

