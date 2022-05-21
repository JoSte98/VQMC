""" Variational Quantum Markov Chain class.
@author: Magdalena and Johannes
"""

import numpy as np
from Helium import Helium
from LHO import LHO
from Hydrogen import Hydrogen

import matplotlib.pyplot as plt

class VQMC:
    """
    Variational Quantum Markov Chain class.
    """

    def __init__(self, num_walkers=400, max_step_length=0.6, num_steps_equilibrate=4000, MC_num_steps=10000,
                 model="Helium", init_alpha=None, Focker_Planck = False):
        """
            Constructor of Variational Quantum Markov Chain class.

            :param num_walkers: [Optional, initial value = 400] (int) Number of individual walkers in Markov Chain.
            :param max_step_length: [Optional, initial value = 0.6] (float) Maximal step length a walker can take in each dimension.
            :param num_steps_equilibrate: [Optional, initial value = 4000] (int) Number of Markov steps to take for equilibration.
            :param MC_num_steps: [Optional, initial value = 10000] (int) Number of Markov steps to take after equilibration to construct measurements.
            :param model: [Optional, initial value = "Helium"] (string) Model (class) which Quantum Markov Chain is performed on,
             including local energy, trial func., etc.
            :param init_alpha: [Optional, initial value = None] Variational parameter (float or np.array, depending on model). In case of 'None'
            the value is set drawn from the model set-up.
            :param Focker_Planck: [Optional, initial value = False] (True/False) Option to use Focker-Planck diffusion equation inspired random walk if True.
        """
        self.model_name = model
        if model == "Helium":
            model = Helium()
        elif model == "Hydrogen":
            model = Hydrogen()
        elif model == "LHO":
            model = LHO()
        else:
            print("Check your model inputs!")
            exit(1)

        self.psi_T = model.trial
        self.energy_L = model.local
        self.energy_L_derivative = model.local_derivative
        self.alpha = model.init_alpha
        self.dimension = model.dimension
        self.derivative_log_trial = model.trial_ln_derivative
        self.derivative_2nd_log_trial = model.trial_ln_2nd_derivative
        
        self.Focker_Planck = Focker_Planck
        if self.Focker_Planck == True:
            self.force = model.force
        else:
            if self.Focker_Planck != False:
                print("Wrong entry of the Focker_Planck parameter! Only 'True' or 'False' are allowed.")
                exit(1)

        if init_alpha is not None:
            self.alpha = init_alpha

        self.num_walkers = num_walkers
        self.chains = [[] for i in range(self.num_walkers)]
        self.initialize_walkers()

        self.max_step_length = max_step_length
        self.energy = []
        
        self.num_tried = 0
        self.num_accepted = 0

        self.num_steps_equilibrate = num_steps_equilibrate
        self.equilibrate(self.num_steps_equilibrate)

        self.MC_num_steps = MC_num_steps

    def reinitialize(self, alpha):
        """
        Reinitialize the whole system for a different variational parameter alpha.
        
        :param alpha: (float or np.array) new variational parameter alpha
        
        return: 0 is successful.
        """
        self.alpha = alpha

        self.chains = [[] for i in range(self.num_walkers)]
        self.initialize_walkers()

        self.energy = []

        self.num_tried = 0
        self.num_accepted = 0

        self.equilibrate(self.num_steps_equilibrate)
        
        return 0

    def initialize_walkers(self):
        """
        Assigns random initial positions of all walkers. The random positions are drawn from Maxwellian distribution
         function centered at 0.

        return: 0 if successful.
        """
        np.random.seed(42)
        init = np.random.normal(loc=0, scale=2, size=(self.num_walkers,self.dimension))
        self.old_psi_squared = []
        
        if self.Focker_Planck==True:
            self.old_force = []
            
        for walker in range(self.num_walkers):
            self.chains[walker].append(init[walker, :])
            self.old_psi_squared.append(self.psi_T(init[walker, :], self.alpha)**2)
            
            if self.Focker_Planck==True:
                self.old_force.append(self.force(init[walker, :], self.alpha))
            

        return 0

    def single_walker_step(self, old_state, old_psi_squared, old_force=None):
        """
        Proposes a new position of the walker and based on Metropolis algorithm either accepts and jumps or stays at the
        original location.

        :param old_state: (np.array) Old state of the walker.
        :param old_psi_squared: (np.array) Corresponding squared trial function value for the old state.
        :param old_force: [Optional, initial value = None] (np.array) None in case of the normal random walk, an old Languir force of the walker for the case of the Focker-Planck approach.
            
        return: Current location of the walker and corresponding squared trial function value. In case of Focker-Planck approach also new Languir force.
        """
        
        self.num_tried += 1
        
        if self.Focker_Planck==False:
            displacement = (2*np.random.rand(self.dimension) - 1)*self.max_step_length
            new_state = old_state + displacement
            new_psi_squared = self.psi_T(new_state, self.alpha)**2
            if old_psi_squared!=0.0:
                p = new_psi_squared/old_psi_squared
            else:
                p = 1.0
            if p >= 1.0:
                self.num_accepted += 1
                return new_state, new_psi_squared
            else:
                q = np.random.random()
                if q < p:
                    self.num_accepted += 1
                    return new_state, new_psi_squared
                else:
                    return old_state, old_psi_squared
            
            
        else:
            displacement = np.random.normal(loc=0.0, scale=1, size=self.dimension) * np.sqrt(self.max_step_length) + old_force *  self.max_step_length/2
            new_state = old_state + displacement
            new_psi_squared = self.psi_T(new_state, self.alpha)**2
            new_force = self.force(new_state, self.alpha)
            if old_psi_squared!=0.0:
                exponent = - np.linalg.norm(new_state - old_state - self.max_step_length * old_force/2 )**2 + np.linalg.norm(old_state - new_state - self.max_step_length * new_force/2 )**2
                p = new_psi_squared/old_psi_squared * np.exp(-exponent/(2*self.max_step_length))
                
            else:
                p=1.0
            if p >= 1.0:
                self.num_accepted +=1
                return new_state, new_psi_squared, new_force
            else:
                q = np.random.random()
                if q < p:
                    self.num_accepted +=1
                    return new_state, new_psi_squared, new_force
                else:
                    return old_state, old_psi_squared, old_force

    def MC_step(self):
        """
        Attempts to shift all the walkers in the model to the next position.
        
        return: 0 if successful.
        """
        if self.Focker_Planck == False:
            for walker in range(self.num_walkers):
                new_state, self.old_psi_squared[walker] = \
                    self.single_walker_step(self.chains[walker][-1], self.old_psi_squared[walker])
                self.chains[walker].append(new_state)
        else:
            for walker in range(self.num_walkers):
                new_state, self.old_psi_squared[walker], self.old_force[walker] = \
                    self.single_walker_step(self.chains[walker][-1], self.old_psi_squared[walker],
                                            old_force=self.old_force[walker])
                self.chains[walker].append(new_state)
        return 0

    def equilibrate(self, num_steps): 
        """
        Gives all walkers a possibility to move num_steps times at the beginning of the simulation in order to equilibrate
        the model.
        
        :param num_steps: (int) Number of attempts to move each walker.
        
        return: 0 if successful.
        """
    
        for i in range(num_steps):
            self.MC_step()
            
        return 0
            
    def energy_mean(self):
        """
        Initializes the system and calculates the mean value of energy for a current alpha of the system.
        
        return: 0 if successful.
        """
        tot_energy = 0
        self.walker_energy = [[] for i in range(self.num_walkers)]
        for i in range(self.MC_num_steps):
            self.MC_step()
            energy = 0
            for walker in range(self.num_walkers):
                E = self.energy_L(self.chains[walker][-1], self.alpha)
                self.walker_energy[walker].append(E)
                energy += E
            self.energy.append(energy/self.num_walkers)
            tot_energy += energy

        print("accepted/tried ratio: ", self.num_accepted / self.num_tried)

        mean_energy_walkers = [np.mean(self.walker_energy[walker]) for walker in range(self.num_walkers)]
        self.variance = np.var(mean_energy_walkers)
        self.uncertainty_energy = np.mean([np.var(self.walker_energy[walker]) for walker in range(self.num_walkers)])
        self.uncertainty_energy_var = np.var([np.var(self.walker_energy[walker]) for walker in range(self.num_walkers)])

        self.chains = [[self.chains[walker][-1]] for walker in range(self.num_walkers)]

        self.expected_energy = tot_energy/((self.MC_num_steps)*self.num_walkers)

        return 0

    def plot_average_local_energies(self):
        """
        Plots the evolution of average local energy of all walkers.
        
        return: 0 if successful.
        """

        plt.plot(range(len(self.energy)), self.energy)
        plt.show()
        return 0
        
    def alpha_energy_dependence(self, stop, steps, start=None, save=True, plot=True):
        """
        Returns lists of alpha and corresponding calculated mean energies of the system.

        :param stop: (float) Maximum value of parameter alpha.
        :param steps: (float) Number of alphas in the interval [start,stop] for which measurement of energy will be done.
        :param start: [optional, initial value = None] (float or None) Start value of alphas, if None -> takes start
         value as current alpha of the model, otherwise takes given start value.
        :param save: [Optional, initial value = True](True/False) Allows saving of the measured alpha-energy dependence
         to a file if True.
        :param plot: [Optional, initial value = True](True/False) Allows plotting of the measured alpha-energy
         dependence if True.
            
        return: Lists of variational parameters alpha, mean energies and their variances.
        """
        
        if start == None:
            alpha_start = self.alpha
        else:
            alpha_start = start
            self.reinitialize(alpha_start)
        alpha_stop = stop
        alphas = np.linspace(alpha_start, alpha_stop, steps)
        mean_energies = np.zeros(steps)
        variances = np.zeros(steps)
        uncertainty = np.zeros(steps)
        uncertainty_var = np.zeros(steps)
        for i in range(alphas.size):
            if i != 0:
                self.reinitialize(alphas[i])
                self.energy_mean()
                mean_energies[i] = self.expected_energy
                variances[i] = self.variance
                uncertainty[i] = self.uncertainty_energy
                uncertainty_var[i] = self.uncertainty_energy_var
            else:
                self.energy_mean()
                mean_energies[i] = self.expected_energy
                variances[i] = self.variance
                uncertainty[i] = self.uncertainty_energy
                uncertainty_var[i] = self.uncertainty_energy_var

        if save:
            self.save_mean_energies(alphas, mean_energies, variances, uncertainty, uncertainty_var)
        if plot:
            self.plot_alpha_energy_dependence(alphas, mean_energies, variances)
                
        return alphas, mean_energies, variances
    
    def save_mean_energies(self, alphas, mean_energies, variances, uncertainty_energy, uncertainty_energy_var,
                           name_of_file=None):
        """
        Saves the lists of alphas, mean energies  and the variances of the local energy and their error estimates
        into a file in the style (alpha mean_energy error_mean var error_var).
        
        :param alphas: (list of floats) List of variational parameters alpha.
        :param mean_energies: (list of floats) List of measured mean energies for given aplhas.
        :param variances: (list of floats) List of corresponding variances of mean energies.
        :param uncertainty_energy: (list of floats) Variance of the local energy.
        :param uncertainty_energy_var: (list of floats) Error estimate of the variance of the local energy.
        :param name_of_file: [Optional, initial value = None](string or None) If string -> name of a file, if None
         generates automatic name of the file itself.

        return: 0 if successful.

        """
        if name_of_file==None:
            name_of_file = "alpha-energy_"+str(alphas[0]).replace(".", "")+"_"+str(alphas[-1]).replace(".", "")+".txt"
            
        with open(name_of_file, "a") as file:
            for i in range(alphas.size):
                file.write("%f %f %f %f %f\n" % (alphas[i], mean_energies[i], variances[i], uncertainty_energy[i],\
                                                 uncertainty_energy_var[i]))

        return 0    
    
    def load_mean_energies(self, name_of_file):
        """
        Loads lists of alphas, energies and variances stored in a file.
        
        :param name_of_file: (string) Name of the file.
        
        return: Lists of variational parameters alpha, mean energies and their variances.
        """
        alphas = []
        mean_energies = []
        variances = []
        with open(name_of_file, "r") as file:
            lines = file.read().split('\n')
            del lines[-1]

            for line in lines:
                alpha, mean, var = line.split(" ")
                alphas.append(float(alpha))
                mean_energies.append(float(mean))
                variances.append(float(var))

        return alphas, mean_energies, variances
    
    def plot_alpha_energy_dependence(self, alphas, mean_energies, variances):
        """
        Plots dependence of the mean values of energy on parameters alpha with variances.
        
        :param alphas: (list of floats) List of variational parameters alpha.
        :param mean_energies: (list of floats) List of measured mean energies for given aplhas.
        :param variances: (list of floats) List of corresponding variances of mean energies.

        return: 0 if successful.
        """
        fig, ax = plt.subplots()

        ax.errorbar(alphas, mean_energies, yerr=np.sqrt(variances), fmt='ro', label="Measurement")

        ax.set_xlabel(r"$\alpha$", fontsize=18)
        ax.set_ylabel(r"Energy", fontsize=18)

        ax.legend(loc="best", fontsize=16)
        ax.grid(visible=True)
        plt.tight_layout()

        plt.show()
        
        return 0









