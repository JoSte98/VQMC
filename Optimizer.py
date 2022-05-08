import Helium as Helium
import Hydrogen as Hydrogen
import LHO as LHO

import matplotlib.pyplot as plt
import numpy as np

class Optimizer:
    def __init__(self, model, max_steplength=0.3, max_steps=50, criterion=1e-3):
        self.max_steplength = max_steplength
        self.max_steps = max_steps
        self.model = model
        self.criterion = criterion
        self.alphas = [self.model.alpha]
        self.energies = []
        self.variances = []

    def gradient(self):
        gradient = 0.0
        for walker in range(self.model.num_walkers):
            for parameter in self.model.chains[walker]:
                gradient += self.model.energy_L(parameter, self.model.alpha) *\
                            self.model.derivative_log_trial(parameter, self.model.alpha)
                gradient -= self.model.expected_energy * self.model.derivative_log_trial(parameter, self.model.alpha)

        gradient *= 2/(self.model.num_walkers * len(self.model.chains[0]))
        return gradient

    def update_alpha(self, step):
        self.model.energy_mean()
        self.energies.append(self.model.expected_energy)
        self.variances.append(self.model.variance)

        new_alpha = self.alphas[-1] - (self.max_steplength/(0+1)) * self.gradient()

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



