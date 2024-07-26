import numpy as np
import matplotlib.pyplot as plt
from loading import get_csv
from dataclasses import dataclass, field

@dataclass
class ScopeNoise:
    csv_path: str
    noisetrace: np.ndarray = field(init=False)
    tInc: float = field(init=False)
    sample_mean: float = field(init=False)
    sample_sd: float = field(init=False)

    def __post_init__(self):
        self.noisetrace, self.tInc = get_csv(self.csv_path)
        #self.noisetrace -= np.min(self.noisetrace)
        #self.noisetrace = np.log(self.noisetrace+1e-7)
        self.t = np.arange(0, self.tInc*len(self.noisetrace), self.tInc)
        self.sample_mean = np.mean(self.noisetrace)
        self.sample_sd = np.std(self.noisetrace, ddof=1)

    def log_error(self):
        pass

    def plot(self):
        plt.figure(figsize=(5,5));
        plt.plot(self.t, self.noisetrace);
        plt.axhline(self.sample_sd, color='black', ls='--')
        plt.axhline(-self.sample_sd, color='black', ls='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.show()
        
    def histogram(self):
        plt.figure(figsize=(10,8))
        plt.hist(self.noisetrace, bins=50)
        plt.axvline(self.sample_mean, color='black', label="mean")
        plt.axvline(self.sample_mean+self.sample_sd, color='red', ls='--', label=f"+sd: {self.sample_sd:0.2e}")
        plt.axvline(self.sample_mean-self.sample_sd, color='red', ls='--', label=f"-sd: {self.sample_sd:0.2e}")
        plt.legend()
        plt.show()


        
def main():
    noise = ScopeNoise("/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/Noise/noise0.csv")
    print(noise.noisetrace)
    print("mean: ", noise.sample_mean)
    print("sd: ", noise.sample_sd)
    noise.histogram()

if __name__ == "__main__":
    main()




