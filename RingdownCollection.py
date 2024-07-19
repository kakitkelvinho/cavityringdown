from RingdownCSV import RingdownCSV
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union
from tqdm import tqdm
from dataclasses import dataclass, field

@dataclass
class RingdownCollection:
    name: str = field(init=False)
    path: str # path to folder which contains a bunch of csv of ringdowns
    ringdowns: list = field(default_factory=list, init=False)
    fitdicts: list = field(init=False)

    def __post_init__(self):
        self.name = self.path.split('/')[-2]
        os.path.expanduser(self.path)
        filenames = [f for f in os.listdir(self.path) if f.endswith('.csv')]
        for filename in tqdm(filenames):
            try:
                ringdown = RingdownCSV(os.path.join(self.path,filename))
            except Exception as e:
                print(f"{e} for {filename}")
                continue
            self.ringdowns.append(ringdown)

    def __getitem__(self, key: Union[str, int]) -> RingdownCSV:
        if isinstance(key, int):
            return self.ringdowns[key]
        elif isinstance(key, str):
            ringdowns_dict = {ringdown.name:ringdown for ringdown in self.ringdowns}
            return ringdowns_dict[key]
        else:
            raise TypeError("Key must either be an integer (index) or a string (name, as key to dictionary).")
            
    def fit_all_ringdowns(self):
        self.fitdicts= [ringdown.fit_by_hand() for ringdown in self.ringdowns]
        return self.fitdicts

    def get_decay_constants(self):
        return {ringdown.name: ringdown.get_decay_constant() for ringdown in self.ringdowns}

    def estimate_finesses(self):
        return {ringdown.name:ringdown.estimate_finesse() for ringdown in self.ringdowns}

    # Plotting methods

    def plot_decay_constants(self, fontsize=20):
        plt.figure(figsize=(10,8))
        plt.style.use('ggplot')
        plt.plot(self.get_decay_constants().values(), 'x')
        plt.xlabel("Ringdowns", fontsize=fontsize)
        plt.ylabel("$\\tau$  decay constant (s)", fontsize=fontsize)
        plt.title(f"Decay times of {self.name}")
        plt.show()

def main():
    ringdowns = RingdownCollection('/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/PA_50/')
    ringdowns['ringdown2'].set_window(13e-6, 20e-6)
    ringdowns.plot_decay_constants()
    #ringdown = RingdownCSV('/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/PA_50/ringdown2.csv')
    #ringdown.set_window(13e-6, 20e-6)
    #ringdown.plot_timetrace()
    #ringdown.plot_logtimetrace()
    #print(ringdown.estimate_finesse())


if __name__ == '__main__':
    main()
