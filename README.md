# VQMC (Variational Quantum Monte Carlo Markov Chain)
This repository implements the Variational Quantum Monte Carlo Markov Chain in order to find ground states 
of externally specified models, including a family of trial functions, specified through variational parameters
$\alpha$, a local energy given by the product $H |\psi\rangle$, where $H$ is the Hamiltonian of the system, and
other derived properties, which are essential for finding the minimum of the energy.

## The models
So far there are 3 different models, the Linear Harmonic Oscillator, the Hydrogen as well as the Helium atom
implemented. For each of those models there is a separate class, specifying the trial function (which can 
also be chosen differently), the corresponding local energy and other properties which will be useful to
express the gradient of the energy in the landscape of the variational parameter. By creating an instance of 
these classes, a starting value for the variational parameter is chosen. Individual models can be choosed by setting the 
parameter 'model='LHO'', 'model='Hydrogen'' or 'model='Helium'' during initialization.

## The Monte Carlo Markov Chain Module
Given a model, we can find the expectation value of the energy via a Monte Carlo Markov Chain Metropolis
kind of algorithm. This can be done via creating an instance of the class VQMC. Here one first can specify
the model (which has to be implemented in a similar manner to the already implemented ones) and various other
hyperparameters of the algorithm, such as the number of independent walkers, the equilibration time as well as
the length of the Markov Chain.
Special caution needs to be taken when choosing the maximum steplength, which represents how far the walkers
are moved in a Markov Chain step. This needs to be reestimated for each model, one may aim for an acceptance
rate of a new state of around 50 %. 
There are two options for creating the Markov Chain, default one is a standart moving of a particle within a 
cube given by the maximum steplength. Second option of Focker-Planck diffusion equation motivated approach can
be allowed by setting the hyperparameter 'Focker-Planck=True' in the initialization.

## The Optimizer Module
The Quantum Monte Carlo Markov Chain Module performs a measurement of the energy for the specified trial 
function for one specific variational parameter. The goal is to find the ground state via minimizing the 
expected energy in the families of possible functions. This can be done with the Optimizer Module. In order
to create an instance of the class, one needs to give an instance of the VQMC class. With the method
Optimizer.find_optimum(), the optimizer will use a specified version of a numerical minimization technique,
in order to find the minimal expected energy, the corresponding variational parameter and thus the best
estimate of the ground state within the family of trial functions.
There are again two options for the numerical minimization technique, both based on gradient descend. First 
is using just the first derivative of energy with respect to the variational parameters and can be switched on
by setting 'gradient_method="1st derivative"' (default value) during initialization of the module. Second is
also incorporating the second derivative of energy with respective to the variational parameters and can be 
switched on by setting 'gradient_method="2nd derivative"'.

## Main
We now want to use the specified modules. Therefore, we go one by one through each model and perform the same 
steps, seeking for the ground state of each of these models.
First we will initialize the VQMC models and then the minimum energy and corresponding paramerers shall be found by
the minimizing algorithm. Thus we call Optimizer.find_optimum() after initializing the Optimazer module.
After that we also plot the dependence of the expected energy on the variational parameter via 
VQMC.alpha_energy_dependence(), in order to visualize the landscape in the variational parameter and the 
minimum.

....
