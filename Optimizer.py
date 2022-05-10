import Helium as Helium
import Hydrogen as Hydrogen
import LHO as LHO

import matplotlib.pyplot as plt
import numpy as np

class Optimizer:
    def __init__(self, model, max_steplength=0.3, max_steps=50, criterion=1e-6, gradient_method = "1st derivative"):
        self.max_steplength = max_steplength
        self.max_steps = max_steps
        self.model = model
        self.criterion = criterion
        self.alphas = [self.model.alpha]
        self.energies = []
        self.variances = []
        self.gradient_method = gradient_method

    def gradient(self):
        gradient11 = 0.0
        gradient12 = 0.0
        gradient21= 0.0
        gradient22= 0.0
        gradient23= 0.0
        gradient24= 0.0
        gradient25= 0.0
        gradient = 0.0
        
        if self.gradient_method=="1st derivative":
            for walker in range(self.model.num_walkers):
                for parameter in self.model.chains[walker]:
                    E_L = self.model.energy_L(parameter, self.model.alpha)
                    log_der1 = self.model.derivative_log_trial(parameter, self.model.alpha)
                    gradient11 += E_L *log_der1
                    gradient12 += log_der1

            gradient = 2/(self.model.num_walkers * len(self.model.chains[0])) * (gradient11 - self.model.expected_energy * gradient12)
            
        elif self.gradient_method=="2nd derivative":
            ### carefull, here so far done only for 1 parameter alpha!!!!! For Lithium must be changed!!! (b will be a vector and H will be a matrix)
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
                    
            b = 2/(self.model.num_walkers * len(self.model.chains[0])) * (gradient11 - self.model.expected_energy * gradient12)
            H = 2/(self.model.num_walkers * len(self.model.chains[0])) * (gradient21 - self.model.expected_energy*gradient22 + 2*(gradient23-self.model.expected_energy*gradient24)-2*b*gradient12+gradient25)
            gradient = b/H           
        return gradient

    def update_alpha(self, step):
        self.model.energy_mean()
        self.energies.append(self.model.expected_energy)
        self.variances.append(self.model.variance)
        
        if self.gradient_method=="1st derivative":
            new_alpha = self.alphas[-1] - (self.max_steplength/(0+1)) * self.gradient()
        elif self.gradient_method=="2nd derivative":
            new_alpha = self.alphas[-1] - self.gradient()
        else:
            print("Gradient method undefined!")

        self.alphas.append(new_alpha)

        #Reinitialize the model
        self.model.reinitialize(new_alpha)

        return 0

    def find_optimum(self, save=True, plot=True):
        for step in range(self.max_steps):
            self.update_alpha(step)
            print("Alpha:", self.alphas[step])
            print("Energy:", self.energies[step])
            print("Variance:", self.variances[step])
            if (np.abs(self.alphas[-1] - self.alphas[-2]) < self.criterion):
                break
            
        self.min_E = min(self.energies)
        self.min_alpha = self.alphas[self.energies.index(self.min_E)]
        
        if save:
            self.save_mean_energies()
        if plot:
            self.plot_alpha_energy_dependence(self.alphas, self.energies, self.variances)

    def save_mean_energies(self, name_of_file=None):
        """

        """
        if name_of_file == None:
            name_of_file = "alpha-energy_" + str(self.model.model_name) + ".txt"

        with open(name_of_file, "a") as file:
            for i in range(len(self.alphas[:-1])):
                file.write("%f %f %f\n" % (self.alphas[i], self.energies[i], self.variances[i]))

        return 0

    def load_mean_energies(self, name_of_file):
        """


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
        Plots dependence of mean value of energy on parameters

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



