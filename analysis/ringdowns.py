import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from dataclasses import dataclass, field
from ringdown import Ringdown, generate_test_timetrace

plt.style.use('seaborn-v0_8-whitegrid')

@dataclass
class Ringdowns:
    name: str = field(default="None provided",)
    ringdowns: list = field(default_factory=list, init=False, repr=False)

    def __getitem__(self, index:int) -> Ringdown:
        return self.ringdowns[index]

    def __len__(self):
        return len(self.ringdowns)

    def __add__(self, ringdown: Ringdown):
        self.ringdowns.append(ringdown)

    def append(self, ringdown: Ringdown):
        self.ringdowns.append(ringdown)

    def __iadd__(self, ringdown: Ringdown):
        self.ringdowns.append(ringdown)
        return self

    def get_taus(self) -> tuple:
        '''Returns a dictionary containing an array of decay time
        and an array for the corresponding error of the decay times'''
        taus = [ringdown.tau for ringdown in self.ringdowns]
        tau_errs = [ringdown.tau_err for ringdown in self.ringdowns]
        return taus, tau_errs

    def get_mean_taus(self) -> float:
        taus, _ = self.get_taus()
        return np.mean(taus)

    def get_std_taus(self) -> float:
        taus, _ = self.get_taus()
        return np.std(taus, ddof=1)

    def get_timetraces(self) -> list:
        '''Returns a list containing all the timetraces'''
        return [ringdown.timetrace for ringdown in self.ringdowns]

    def get_cropnormtimetraces(self) -> list:
        '''Returns a list containing all the cropped and normalized of timetraces'''
        return [ringdown.cropnormtimetrace for ringdown in self.ringdowns]

    def get_logtimetraces(self) -> list:
        '''Returns a list containing all the log of timetraces'''
        return [ringdown.logtimetrace for ringdown in self.ringdowns]


    ## Plotting methods

    def plot_taus(self, fontsize=18):
        taus, tau_errs = self.get_taus()
        mean_taus = np.mean(taus)
        std_taus = np.std(taus, ddof=1)

        plt.figure(figsize=(10,8), layout='constrained')

        plt.errorbar(x=np.arange(0, len(taus)), y=np.array(taus)/1e-6, yerr=np.array(tau_errs)/1e-6, capsize=0.5, marker='x', ls='', ecolor='black')
        plt.axhline(mean_taus/1e-6, label=f"$\\tau$ mean)/1e-6: {mean_taus:.2e} s", color='gray', linestyle='--')
        plt.axhspan((mean_taus + std_taus)/1e-6, (mean_taus - std_taus)/1e-6, fill=0, ls='--', color='red', label=f"$\\sigma$ = {std_taus:.2e} s")
        plt.xlabel("Measurement number", fontsize=fontsize-2)
        plt.ylabel("Decay time ($\\mu$s)", fontsize=fontsize-2)
        plt.title(f"Decay time constant $\\tau$ across runs", fontsize=fontsize+5)    
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', axis='both')
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
        plt.xticks(np.arange(0, len(taus)+1, 5))
        plt.show()


def main():
    n = 50
    a = np.random.normal(0.7, 0.001, n)
    taus = np.random.normal(1.9e-6, 1e-8, n)
    c = 0.1
    noise_sd = np.mean(a)/50
    ringdowns = Ringdowns(name="test")
    for i in range(n):
        t, trace = generate_test_timetrace(a[i], taus[i], c, noise_sd) 
        ringdown = Ringdown(timetrace=trace, t=t)
        ringdowns += ringdown

    print(ringdowns.get_std_taus())

if __name__ == '__main__':
    main()
