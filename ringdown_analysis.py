import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import re
from scipy.optimize import curve_fit
import pandas as pd


# define the exponential function

def exponential_decay(t, tau):
    return np.exp(-t/tau)

experiment = '20240708_Ringdown' # this sort of depends on where your specific experimental data is
all_data = '~/LabInnsbruck/WindowsData' # please change depending on where you put your data dir
data_dir = os.path.expanduser(os.path.join(all_data, experiment))
print(os.path.join(all_data,experiment) == '~/LabInnsbruck/WindowsData/20240708_Ringdown')

# look inside the experiment directory

csv_filenames = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
decays = []
tInc = 5e-10 # maybe not have to hardcode this?

for filename in csv_filenames:
    fullpath = os.path.join(data_dir, filename)
    df = pd.read_csv(fullpath)
    # print(float(df.head(0).to_string().split(',')[-2].split('=')[-1].split('s')[0]))
    decay = df['CH1(V)']
    decays.append(decay)


# fix offset and normalize
decays = [decay - np.min(decay) for decay in decays]
# decays = [decay/np.max(decay) for decay in decays] # I don't want to normalize actually

## function to crop
def crop_decay(t, decay, t0, t1):
    mask = np.logical_and(t >= t0, t<=t1)
    return decay[mask]

def crop_time(t, t0, t1):
    mask = np.logical_and(t >= t0, t<=t1)
    return t[mask]



## Supply a window to crop 
n = len(decays[0])
t = np.arange(0, n*tInc, tInc)
t0 = 12e-6
t1 = 16e-6
#
## Crop the traces
t_cropped = crop_time(t, t0, t1)
t_cropped_offset = t_cropped - t_cropped[0]
cropped_decays = [crop_decay(t, decay, t0, t1) for decay in decays]
log_decays = [np.log(decay) for decay in cropped_decays]

# then you can fit
for decay in log_decays:
    p_fitted = np.polynomial.Polynomial.fit(t_cropped_offset, decay, deg=1)
    print(p_fitted, p_fitted.coef)


# plot log data
plt.figure()
for i, decay in enumerate(log_decays):
    plt.plot(t_cropped_offset, decay, alpha=0.2)
plt.plot(t_cropped_offset, p_fitted(t_cropped_offset), label='last fit')
plt.savefig(f'{experiment}_log_decay.png')

# plot data
#
plt.figure()
for i, decay in enumerate(decays):
    if i%5 == 0:
        plt.subplot(211)
        plt.plot(t, decay)
        plt.xlim([t0, t1])
        plt.subplot(212)
        plt.semilogy(t, decay)
        plt.xlim([1e-5, 2e-6])
plt.savefig(f'{experiment}_decays.png', dpi=300)


