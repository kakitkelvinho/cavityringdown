import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.optimize import curve_fit
import pands as pd


# define the exponential function

def exponential_decay(t, tau):
    return np.exp(-t/tau)

experiment = '20240708_Ringdown' # this sort of depends on where your specific experimental data is
all_data = '~/LabInnsbruck/WindowsData' # please change depending on where you put your data dir
data_dir = os.path.expanduser(os.path.join(all_data, experiment))
# look inside the experiment directory

csv_filenames = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
decays = []

for filename in csv_filenames:
    file_path = os.path.join(data_dir, filename)
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skips header row
        decay = []
        for row in reader:
            if row[0]:
                decay.append(float(row[0]))
        decays.append(decay)

## Supply a window to crop 
t_0 = 1e-6
t_1 = 6e-6

# plot data

plt.figure()
for decay in decays:
    plt.plot(decay)
plt.savefig(f'{experiment}_decays.png', dpi=300)

