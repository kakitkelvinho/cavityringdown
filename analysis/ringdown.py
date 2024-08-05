import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import log

plt.style.use('seaborn-v0_8-whitegrid')

@dataclass
class Ringdown:
    timetrace: np.ndarray = field(repr=False)
    t: np.ndarray = field(repr=False)
    pa_angle: int = field(default=0, init=False)
    t0: float = field(default=0., init=False, repr=False) # start of window
    window: float = field(default=1.5e-6, init=False, repr=False) # size of window

    t_crop: np.ndarray = field(repr=False, init=False)
    mask: np.ndarray = field(repr=False, init=False)
    cropnormtimetrace: np.ndarray = field(repr=False, init=False)
    logtimetrace: np.ndarray = field(repr=False, init=False)

    def __post_init__(self):
        # ensure that input is numpy array
        pass

    def fit(self, t0=None, window=None, p0=[0.2, 1e-6, 0.], plot=True,):
        '''Given t0 and a window, 
        1. crops time trace
        2. takes the log
        3. fits to it
        returns: popt, pcov -- the fit parameters and covariance matrix
        '''
        if t0 == None:
            t0 = self.t0
        if window == None:
            window = self.window

        self.t_crop, self.mask, self.cropnormtimetrace, self.logtimetrace = self.create_logtimetrace(t0, window)

        # fit timetrace
        popt, pcov = curve_fit(self.fit_func, xdata=self.t_crop, ydata=self.logtimetrace, p0=p0)
        
        # get residuals
        residuals = self.logtimetrace - self.fit_func(self.t_crop, *popt)
        # calculate rmse
        rmse = np.sqrt(np.sum(residuals*residuals)/(len(self.logtimetrace)-3))
        print(f"rmse: {rmse}")

        ### EDIT LATER
        if plot:
            self.plot_fit(popt, pcov)
        return popt, pcov

    def fit_func(self, t, a, tau, c):
        return np.log(a*np.exp(-t/tau) + c)

    def create_logtimetrace(self, t0, window, offset=1e-12):
        mask = np.logical_and(self.t >= t0, self.t <= t0+window)
        timetrace = self.timetrace[mask] - np.min(self.timetrace) + offset
        cropnormtimetrace = timetrace / np.max(timetrace)
        logtimetrace = np.log(cropnormtimetrace)
        t_crop = self.t[mask]
        t_crop -= t_crop[0] # to start at t=0
        return t_crop, mask, cropnormtimetrace, logtimetrace

    def exp_residual(self, t0=None, window=None, p0=[0.2, 1e-6, 0.]):
        if t0 == None:
            t0 = self.t0
        if window == None:
            window = self.window
        popt, _ = self.fit(t0, window, p0, plot=False)
        residual_from_exp = self.cropnormtimetrace - (popt[0]*np.exp(-self.t_crop/popt[1]) - popt[1])
        return np.std(residual_from_exp, ddof = 1)
        

    def plot_fit(self, popt, pcov):
        '''
        Plots the ringdown (1), log ringdown + fit (2) and residuals (3), 
        and the cropped ringdown + fit (4) and residual (5)
        |---------------|
        |       |       |
        |       |   2.  |        
        |       |       |
        |       |-------|
        |       |   3.  |
        |   1.  |-------|
        |       |       |
        |       |   4.  |
        |       |       |
        |       |-------|
        |       |   5.  |
        |---------------|
        '''
        delta_tau = np.sqrt(pcov[1][1])
        fig = plt.figure(figsize=(15,8))
        fig.suptitle("Ringdowns and Fits")
        gs = fig.add_gridspec(2, 2, width_ratios=(1,1), height_ratios=(1,1))

        ax1 = fig.add_subplot(gs[:,0])
        ax1.plot(self.t/1e-6, self.timetrace)
        ax1.axvline(self.t0/1e-6, color='black')
        ax1.axvline((self.t0+self.window)/1e-6, color='black')
        ax1.set_xlabel("Time ($\\mu$s)")
        ax1.set_ylabel("Intensity (V)")
        ax1.set_title("Recorded Timetrace")
        
        right_gs = gs[:, 1].subgridspec(4, 1, height_ratios=[3,1,3,1])

        ax23 = fig.add_subfigure(gs[0,1])
        ax23.subplots_adjust(hspace=0)

        ax2 = ax23.add_subplot(right_gs[0,0])
        ax2.set_title("Log of intensity")
        ax2.plot(self.t_crop/1e-6, self.logtimetrace, label="log", markersize=1, alpha=0.4)
        ax2.plot(self.t_crop/1e-6, self.fit_func(self.t_crop, *popt), label=f"$\\tau$: {popt[1]:0.2e} ({delta_tau:0.1e})s")
        ax2.tick_params(labelbottom=False)
        ax2.set_ylabel("Intensity (V)")
        ax2.legend()

        ax3 = ax23.add_subplot(right_gs[1,0], sharex=ax2)
        ax3.plot(self.t_crop/1e-6, self.logtimetrace - self.fit_func(self.t_crop, *popt))
        ax3.set_ylabel("Residual")
        ax3.set_xlabel("Time ($\\mu$s)")
        
        ax45 = fig.add_subfigure(gs[1,1])
        ax45.subplots_adjust(hspace=0)

        ax4 = ax45.add_subplot(right_gs[2,0])
        ax4.set_title("Cropped and Normalized Timetrace")
        ax4.plot(self.t_crop, self.cropnormtimetrace, markersize=1, alpha=0.4, label="cropped and normalized")
        ax4.plot(self.t_crop, popt[0]*np.exp(-self.t_crop/popt[1])+popt[2], label="fit w/ above params")
        ax4.tick_params(labelbottom=False)
        ax4.set_ylabel("Intensity (V)")
        ax4.tick_params(labelbottom=False)
        ax4.legend()


        ax5 = ax45.add_subplot(right_gs[3,0], sharex=ax4)
        ax5.plot(self.t_crop, popt[0]*np.exp(-self.t_crop/popt[1])+popt[2] - self.cropnormtimetrace)
        ax5.set_ylabel("Residual")
        ax5.set_xlabel("Time ($\\mu$s)")
        
        plt.show()


#
#        plt.subplot(121)
#        plt.plot(self.t, self.timetrace)
#        plt.subplot(222)
#        plt.plot(t_crop, logtimetrace)
#        plt.plot(t_crop, self.fit_func(t_crop, *popt))
#        plt.subplot(224)
#        plt.plot(t_crop, self.cropnormtimetrace)
#        plt.plot(t_crop, popt[0]*np.exp(-t_crop/popt[1])+popt[2])
#        plt.legend()
#        plt.tight_layout()
#        plt.show()
        
def main():
    t = np.arange(0, 2e-6, 2.5e-10)
    a, tau, c = [0.4, 1.8e-6, 0.03]
    trace = a*np.exp(-t/tau) - c
    noise_sd = a/100
    noise = np.random.normal(0, noise_sd, len(trace))
    trace += noise
    ringdown = Ringdown(timetrace=trace, t=t)
    print(f"injected noise: {noise_sd}")

    popt, pcov = ringdown.fit()
    delta_a, delta_tau, delta_c = np.sqrt(np.diag(pcov))




if __name__ == '__main__':
    main()



        

