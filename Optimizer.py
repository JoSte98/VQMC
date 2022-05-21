""" Optimizer for Variational Quantum Markov Chain class.
@author: Magdalena and Johannes
"""

import matplotlib.pyplot as plt
import numpy as np

class Optimizer:
    """
    Optimization class for the Variational Quantum Markov Chain.
    """
    def __init__(self, model, steplength=0.3, max_steps=50, criterion=1e-6, gradient_method="1st derivative"):
        """
            Constructor for the Optimization class for the Variational Quantum Markov Chain.

            :param model: Variational Quantum Markov Chain class (initialized).
            :param steplength: [Optional, initial value = 0.3](float) Step length of the gradient descent method.
            :param max_steps: [Optional, initial value = 50](int) Maximum of minimization steps to take.
            :param criterion: [Optional, initial value = 1e-6](float) Criterion for stopping minimization procedure. If new and old alpha differ less
             than criterion => stop.
            :param gradient_method: [Optional, initial value = "1st derivative"]{"1st derivative", "2nd derivative"} Choose between minimization methods (normal
             gradient descent and minimization taking the 2nd derivative into account).
        """
        self.steplength = steplength
        self.max_steps = max_steps
        self.model = model
        self.criterion = criterion
        self.alphas = [self.model.alpha]
        self.energies = []
        self.variances = []
        self.gradient_method = gradient_method

    def gradient(self):
        """
        Gradient at the current position of the variational parameter.
        If the method is given by "1st derivative" it calculates the pure gradient, if 2nd derivative" it calculates
        the whole minimization step.

        :return: Gradient.
        """
        gradient11 = 0.0
        gradient12 = 0.0
        gradient21= 0.0
        gradient22= 0.0
        gradient23= 0.0
        gradient24= 0.0
        gradient25= 0.0
        gradient = 0.0
        
        if self.gradient_method == "1st derivative":
            for walker in range(self.model.num_walkers):
                for parameter in self.model.chains[walker]:
                    E_L = self.model.energy_L(parameter, self.model.alpha)
                    log_der1 = self.model.derivative_log_trial(parameter, self.model.alpha)
                    gradient11 += E_L *log_der1
                    gradient12 += log_der1

            gradient = 2/(self.model.num_walkers * len(self.model.chains[0])) *\
                        (gradient11 - self.model.expected_energy * gradient12)
            
        elif self.gradient_method == "2nd derivative":
            ### carefull, here so far done only for 1 parameter alpha!!!!! For more cvariational parameters must be changed!!!
            ### (g will be a vector and H will be a matrix)
            for walker in range(self.model.num_walkers):
                for parameter in self.model.chains[walker]:
                    E_L = self.model.energy_L(parameter, self.model.alpha)
                    log_der1 = self.model.derivative_log_trial(parameter, self.model.alpha)
                    log_der2 = self.model.derivative_2nd_log_trial(parameter, self.model.alpha)
                    der_E_L = self.model.energy_L_derivative(parameter, self.model.alpha)
                    gradient11 += E_L *log_der1
                    gradient12 += log_der1
                    gradient21 += E_L * log_der2
                    gradient22 += log_der2
                    gradient23 += E_L * log_der1**2
                    gradient24 += log_der1**2
                    gradient25 += log_der1 * der_E_L
                    
            b = 2/(self.model.num_walkers * len(self.model.chains[0])) *\
                (gradient11 - self.model.expected_energy * gradient12)
            H = 2/(self.model.num_walkers * len(self.model.chains[0])) *\
                (gradient21 - self.model.expected_energy * gradient22 +
                 2*(gradient23-self.model.expected_energy*gradient24)-2*b*gradient12+gradient25)
            gradient = b/H

        return gradient

    def update_alpha(self):
        """
        Updates the current value of the variational parameter alpha, taking into account the method and
        the gradient.

        :return: 0 of successful.
        """
        self.model.energy_mean()
        self.energies.append(self.model.expected_energy)
        self.variances.append(self.model.variance)
        
        if self.gradient_method=="1st derivative":
            new_alpha = self.alphas[-1] - self.steplength * self.gradient()
        elif self.gradient_method=="2nd derivative":
            new_alpha = self.alphas[-1] - self.gradient()
        else:
            print("Gradient method undefined!")
            exit(-1)

        self.alphas.append(new_alpha)

        #Reinitialize the model with the new alpha parameters
        self.model.reinitialize(new_alpha)

        return 0

    def find_optimum(self, save=True, plot=True):
        """
        Finds the optimal value of the variational parameter for the model.

        :param save: [Optional, default = True] (boolean) Enables saving the alpha-energy history of the minimization procedure.
        :param plot: [Optional, default = True] (boolean) Enables plotting the alpha-energy history of the minimization procedure.

        :return: 0 if successful.
        """

        for step in range(self.max_steps):
            self.update_alpha()
            print("Alpha:", self.alphas[step])
            print("Energy:", self.energies[step])
            print("Variance:", self.variances[step])
            if (np.abs(self.alphas[-1] - self.alphas[-2]) < self.criterion):
                break
            
        self.min_E = min(self.energies)
        self.min_alpha = self.alphas[self.energies.index(self.min_E)]
        self.min_variance = self.variance[self.energies.index(self.min_E)]
        
        if save:
            self.save_mean_energies()
        if plot:
            self.plot_alpha_energy_dependence(self.alphas, self.energies, self.variances)

        return 0

    def save_mean_energies(self, name_of_file=None):
        """
        Saves all alphas and the corresponding energies of the Markov chain as well as the variance in file.

        :param name_of_file: [optional, default: None] Name of the file (default will save file under model name).

        :return: 0 if successful.
        """
        if name_of_file is None:
            name_of_file = "alpha-energy_" + str(self.model.model_name) + ".txt"

        with open(name_of_file, "a") as file:
            for i in range(len(self.alphas[:-1])):
                file.write("%f %f %f\n" % (self.alphas[i], self.energies[i], self.variances[i]))

        return 0

    def load_mean_energies(self, name_of_file):
        """
        Load the alphas, energies and variances out of file.

        :param name_of_file: (string) Name of the file.

        :returns:
         - alphas - (list of floats) List of variational parameters.
         - energies - (list of floats) List of corresponding energies.
         - variances - (list of floats) List of corresponding variances.
        """
        alphas = []
        energies = []
        variances = []
        with open(name_of_file, "r") as file:
            lines = file.read().split('\n')
            del lines[-1]

            for line in lines:
                alpha, mean, variance = line.split(" ")
                alphas.append(float(alpha))
                energies.append(float(mean))
                variances.append(float(variance))

        return alphas, energies, variances

    def plot_alpha_energy_dependence(self, alphas, energies, variances):
        """
        Plots dependence of mean value of energy and variance on variational parameters.

        :param alphas: (list of floats) List of variational parameters.
        :param energies: (list of floats) List of corresponding energies.
        :param variances: (list of floats) List of corresponding variances.

        :return: 0 if successful.
        """
        fig, ax = plt.subplots()

        ax.errorbar(range(len(alphas[:-1])), energies, yerr=np.sqrt(variances), fmt='ro', label="Measurement")

        ax.set_xticks(range(len(alphas[:-1])), alphas[:-1])

        ax.set_xlabel(r"Steps", fontsize=18)
        ax.set_ylabel(r"Energy", fontsize=18)

        ax.legend(loc="best", fontsize=16)
        ax.grid(visible=True)
        plt.tight_layout()

        plt.show()

        return 0



