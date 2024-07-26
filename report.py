import numpy as np
import matplotlib.pyplot as plt
from analysis.ringdowncsv import RingdownCSV
from analysis.ringdowncollection import RingdownCollection
from analysis.loading import get_csv
import os

def main():
    folder_path = '/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/' # define folder containing ringdowns
    angles = os.listdir(folder_path)
    angles = [angle for angle in angles if 'PA' in angle]

    print(angles)
    scope_err = 7e-4 # error of the scope

    # generate a dictionary of Ringdown Collections
    #angles_ringdown = {angle:RingdownCollection(os.path.join(folder_path, angle)) for angle in angles}

    pa0 = angles[-1]
    ringdowncollection = RingdownCollection(os.path.join(folder_path, pa0))
    finesses = []
    error_f = []
    sd_ests = []
    sd_fits =[]
    
    for ringdown in ringdowncollection:
        finesse, error = ringdown.estimate_finesse()
        finesses.append(finesse)
        error_f.append(error)
        sd_est, sd_fit = ringdown.compare_errors()
        sd_ests.append(sd_est)
        sd_fits.append(sd_fit)

    plt.figure(figsize=(8,8))
    plt.subplot(211)
    plt.errorbar(np.arange(0,len(finesses)), finesses, yerr=error_f,
                 capsize=2, ls='', marker='x')
    plt.subplot(212)
    plt.plot(sd_ests, label="s.d. from prop.")
    plt.plot(sd_fits, label="error from fit")
    plt.legend()
    plt.show()

    print(ringdowncollection)



if __name__ == "__main__":
    main()
