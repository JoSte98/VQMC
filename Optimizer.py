import Helium as Helium
import Hydrogen as Hydrogen
import LHO as LHO

class Optimizer:
    def __init__(self, model, max_steplength=1.0, max_steps=100):
        self.max_steplength = max_steplength
        self.max_steps = max_steps
        self.model = model
        self.alphas = [self.model.alpha]
        self.energies = []
        self.variance = []


    def gradient(self):
        gradient = 0.0
        for walker in range(self.model.num_walkers):
            for parameter in self.model.chains[walker]:
                gradient += self.model.energy_L(parameter, self.model.alpha) * self.model.derivative_log_trial(parameter,
                                                                                                               self.model.alpha)
                gradient -= self.model.expected_energy * self.model.derivative_log_trial(parameter, self.model.alpha)
        gradient *= 2/(self.model.num_walkers * len(self.model.chains[0]))

        return gradient

    def update_alpha(self, step):
        self.model.get_energy_mean()
        self.energies.append(self.model.expected_energy)

        #self.variance.append

        new_alpha = self.alphas[-1] - (self.max_steplength/(0+1)) * self.gradient()

        self.alphas.append(new_alpha)

        #Reinitialize the model
        self.model.reinitialize(new_alpha)

        return 0

    def find_optimum(self):
        for step in range(self.max_steps):
            self.update_alpha(step)
            print("Alpha:", self.alphas[step])
            print("Energy:", self.energies[step])



