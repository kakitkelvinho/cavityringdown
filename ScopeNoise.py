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
        self.sample_mean = np.mean(self.noisetrace)
        self.sample_sd = np.std(self.noisetrace, ddof=1)
        

        
        



