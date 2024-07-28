import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from analysis.ringdowncsv import RingdownCSV
from analysis.ringdowncollection import RingdownCollection
from analysis.loading import get_csv
import time
import os

plt.style.use('seaborn-v0_8-whitegrid')
folder_path = '/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/' # define folder containing ringdowns
scope_err = 7e-4 # error of the scope

def one_angle_log_residual(angle_name: str, projection='2d', save=False, show=True):
    ringdowncollection = RingdownCollection(os.path.join(folder_path, angle_name))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / len(ringdowncollection)) for i in range(len(ringdowncollection))]
    fig = plt.figure(figsize=(10,10))
    if projection=='3d':
        ax = fig.add_subplot(projection='3d')
        for i, ringdown in enumerate(ringdowncollection):
            a = ringdown.handfitdict["A"]
            b = ringdown.handfitdict["B"]
            ax.plot(ringdown.croptime_offset, ringdown.logtimetrace, zs=2*i, zdir='y', label=ringdown.name, color=colors[i], marker='.', markersize=1, ls='')
            ax.plot(ringdown.croptime_offset, a+b*ringdown.croptime_offset, zs=2*i, zdir='y', color='black')
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_zlabel("Log of intensity")
        ax.set_ylabel("Trials")
    elif projection=='2d':
        gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        for i, ringdown in enumerate(ringdowncollection):
            ax1.plot(ringdown.croptime_offset, ringdown.logtimetrace, color=colors[i], label=ringdown.name, marker='.', markersize=1, ls='')
            a = ringdown.handfitdict['A']
            b = ringdown.handfitdict['B']
            ax2.plot(ringdown.croptime_offset, ringdown.logtimetrace-a-b*ringdown.croptime_offset, marker='.', markersize=1, ls='')
            ax2.axhline(0, ls='--', color='gray')
        ax2.set_xlabel("Time (s)")
        ax1.set_ylabel("Log of intensity")
        ax2.set_ylabel("Residuals")
        fig.suptitle(f"Ringdowns with {angle_name}")
    else:
        print("Please state whether projection is 2d or 3d.") 
    if show:
        plt.show()
    if save:
        plt.savefig(f"./results/log_residual/ringdowns_{angle_name}.png")

def one_angle_finesse(angle_name:str, show=True, save=False):
    ringdowncollection = RingdownCollection(os.path.join(folder_path, angle_name))
    finesses = np.array([ringdown.estimate_finesse() for ringdown in ringdowncollection])
    taus = [ringdown.get_decay_constant() for ringdown in ringdowncollection]
    taus_error = [ringdown.handfitdict['delta_B']*(ringdown.get_decay_constant())**2 for ringdown in ringdowncollection]
    compares = np.array([ringdown.compare_errors() for ringdown in ringdowncollection])
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(221)
    plt.errorbar(np.arange(0,len(finesses)), finesses[:,0], yerr=finesses[:,1], ls='',
                 ecolor='black', capsize=1, capthick=1)
    plt.title("Finesses")

    plt.subplot(222)
    plt.errorbar(np.arange(0, len(taus)), taus, yerr=taus_error, ls='',
                 ecolor='black', capsize=1, capthick=1)
    plt.title("Decay times")

    plt.subplot(223)
    plt.plot(compares[:,0], label="estimate")
    plt.plot(compares[:,1], label="from fit")
    plt.legend()
    plt.title("Error comparison")

    plt.subplot(224)
    plt.hist(finesses[:,0], bins=30)
    plt.title("Finesse bin")

    if show:
        plt.show()
    if save:
        plt.savefig(f'./results/finesse/{angle_name}.png')

    

def all_angle_plots(show=True, save=False):
    """To plot in 3D the ringdowns, in the style shown in Dupre (2015)"""
    # collect all the ringdowns
    angles = os.listdir(folder_path)
    angles = [angle for angle in angles if 'PA' in angle]
    ringdown_angles = {int(angle.split('_')[-1]): f"{folder_path}/{angle}/ringdown1.csv" for angle in angles}
    ringdown_angles = sorted(ringdown_angles.items())
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    cmap =  plt.get_cmap('tab20')
    colors = [cmap(i / len(angles)) for i in range(len(angles))]

    
    for key, value in ringdown_angles:
        ringdown = RingdownCSV(value)
        ax.plot(ringdown.croptime_offset, ringdown.logtimetrace, zs=key, zdir='y', marker='.', markersize=1, ls='', color=colors[int(key/10)])
    ax.set_xlabel("Time (s)")
    ax.set_zlabel("Log of intensity")
    ax.set_ylabel("PA angles (degrees)")
    fig.suptitle(f"Ringdowns at different PA angles")
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig("./results/ringdowns_at_diff_PA_angles.png")



def main():
    tick = time.time()
    angles = os.listdir(folder_path)
    angles = [angle for angle in angles if 'PA' in angle]
    for angle in angles:
        one_angle_log_residual(angle, projection='2d', save=True, show=False)
        one_angle_finesse(angle, show=False, save=True)
    all_angle_plots(show=False, save=True)
    tock = time.time()
    print(f"Elapsed time: {(tock-tick)/60} mins")





if __name__ == "__main__":
    main()
