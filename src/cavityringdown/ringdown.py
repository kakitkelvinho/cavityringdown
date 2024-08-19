import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from scipy.optimize import curve_fit
import os
from typing import Optional

plt.style.use('seaborn-v0_8-whitegrid')

@dataclass
class Ringdown:
    timetrace: np.ndarray = field(repr=False)
    t: np.ndarray = field(repr=False)
    pa_angle: int = field(default=0, init=False)
    t0: Optional[float] = field(default=None, init=True, repr=False) # start of window
    window: float = field(default=1.5e-6, init=True, repr=False) # size of window

    t_crop: np.ndarray = field(repr=False, init=False)
    mask: np.ndarray = field(repr=False, init=False)
    cropnormtimetrace: np.ndarray = field(repr=False, init=False)
    logtimetrace: np.ndarray = field(repr=False, init=False)

    tau: float = field(default=0., init=False) # decay time
    tau_err: float = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if self.t0 == None:
            self.t0 = self.findpeak()
        popt, pcov = self.fit()
        self.tau = popt[1]
        self.tau_err = np.sqrt(np.diag(pcov))[1]

    def findpeak(self, rolloff:float=0.5e-6):
        # cropping the basic attributes or taking the log
        return self.t[np.argmax(self.timetrace[:round(len(self.timetrace)/2)])] + rolloff # find the max that lies in the first half of the array



    def fit(self, t0=None, window=None, p0=[0.2, 1e-6, 0.], plot=False, show=False, additional_trace=None):
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
        else:
            self.t0 = t0
            self.window = window

        self.t_crop, self.mask, self.cropnormtimetrace, self.logtimetrace = self.create_logtimetrace(t0, window)

        # fit timetrace
        popt, pcov = curve_fit(self.fit_func, xdata=self.t_crop, ydata=self.logtimetrace, p0=p0)
        
        # get residuals
        residuals = self.logtimetrace - self.fit_func(self.t_crop, *popt)
        # calculate rmse
        #rmse = np.sqrt(np.sum(residuals*residuals)/(len(self.logtimetrace)-3))

        ### EDIT LATER
        if plot:
            fig = self.plot_fit(popt, pcov, show=show, additional_trace=additional_trace)
            return fig
        return popt, pcov

    def fit_func(self, t, a, tau, c):
        return np.log(a*np.exp(-t/tau) + c)

    def create_logtimetrace(self, t0, window, offset=1e-12):
        '''Generates log of timetrace, 
        returns cropped time, a mask, 
        the cropped and normalized timetrace and log of timetrace'''
        mask = np.logical_and(self.t >= t0, self.t <= t0+window)
        # check whether any value in the *cropped* timetrace is less than zero
        # if yes offset the timetrace, because otherwise it messes up fit
        timetrace = self.timetrace[mask] - np.min(self.timetrace) + offset
        #if np.any(timetrace) <= 0:
        #    timetrace -= np.min(timetrace) + offset
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
        

    def plot_fit(self, popt, pcov, path='', name='plot_fit.png', additional_trace=None, show=False):
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
        fig.suptitle("Ringdowns and Fits", fontsize=20)
        subfigs = fig.subfigures(1,2, wspace=0.01)
        left = subfigs[0]
        #gs = fig.add_gridspec(2, 2, width_ratios=(1,1), height_ratios=(1,1))


        ax1 = left.add_subplot(111)
        ax1.plot(self.t/1e-6, self.timetrace, color='tab:blue', marker='.', markersize=1, ls='none', label='raw timetrace')
        ax1.axvline(self.t0/1e-6, color='black', ls='--', label='start time')
        ax1.axvline((self.t0+self.window)/1e-6, ls='--', color='black', label='end time')
        ax1.set_xlabel("Time ($\\mu$s)")
        ax1.set_ylabel("Intensity (V)")
        ax1.set_title("Recorded Timetrace")
        if additional_trace is not None:
            color = 'tab:purple'
            ax11 = ax1.twinx()
            ax11.set_ylabel("Intensity (V)", color=color)
            ax11.plot(self.t/1e-6, additional_trace, color=color, marker='.', markersize=1, ls='none', label="ref.", alpha=0.5)
            ax11.tick_params(axis='y', labelcolor=color)
            ax11.grid(False)
        #ax1.legend(markerscale=10)
        left.legend(markerscale=10, borderaxespad=8.)
            

        
        right = subfigs[1].subfigures(2, 1, height_ratios=[1,1])
        righttop = right[0].add_gridspec(2, 1, height_ratios=[3,1])
        rightbottom = right[1].add_gridspec(2,1, height_ratios=[3,1])



        right[0].subplots_adjust(hspace=0)
        ax2 = right[0].add_subplot(righttop[0])
        ax2.set_title("Log of intensity")
        ax2.plot(self.t_crop/1e-6, self.logtimetrace, label="log", marker='.', ls='none', markersize=1, alpha=0.6, color='tab:red')
        ax2.plot(self.t_crop/1e-6, self.fit_func(self.t_crop, *popt), label=f"$\\tau$: {popt[1]:.2e}s, err: {delta_tau:.2e} s", color='darkmagenta')
        ax2.tick_params(labelbottom=False)
        ax2.set_ylabel("Intensity (V)")
        ax2.legend(markerscale=10)

        ax3 = right[0].add_subplot(righttop[1], sharex=ax2)
        residual = self.fit_func(self.t_crop, *popt) - self.logtimetrace
        mean_res = np.mean(residual)
        std_res = np.std(residual, ddof=1)
        ax3.plot(self.t_crop/1e-6, residual, marker='.', markersize=1, alpha=0.8, ls='', color='tab:red')
        ax3.axhspan(mean_res + std_res, mean_res - std_res, fill=0, ls='--', color='black', label=f"$\\sigma:$ {std_res:.1e}")
        ax3.legend(frameon=True, facecolor='white', loc='lower left', markerscale=10)
        ax3.set_ylabel("Residual")
        ax3.set_xlabel("Time ($\\mu$s)")
        
        right[1].subplots_adjust(hspace=0)
        print(popt[0])
        ax4 = right[1].add_subplot(rightbottom[0])
        ax4.set_title("Cropped and Normalized Intensity")
        ax4.plot(self.t_crop/1e-6, self.cropnormtimetrace, marker='.', ls='', markersize=1, alpha=0.6, label="cropped and normalized", color='tab:green')
        ax4.plot(self.t_crop/1e-6, popt[0]*np.exp(-self.t_crop/popt[1])+popt[2], label="fit w/ above params", color='darkmagenta')
        ax4.tick_params(labelbottom=False)
        ax4.set_ylabel("Intensity (V)")
        ax4.tick_params(labelbottom=False)
        ax4.legend(markerscale=10)


        ax5 = right[1].add_subplot(rightbottom[1], sharex=ax4)
        residual = self.cropnormtimetrace - (popt[0]*np.exp(-self.t_crop/popt[1])+popt[2])
        mean_res = np.mean(residual)
        std_res = np.std(residual, ddof=1)
        ax5.plot(self.t_crop/1e-6, residual, marker='.', ls='', markersize=1, alpha=0.8, color='tab:green')
        ax5.axhspan(mean_res + std_res, mean_res - std_res, fill=0, ls='--', color='black', label=f"$\\sigma:$ {std_res:.1e}")
        ax5.legend(frameon=True, facecolor='white', loc='lower left', markerscale=10)
        ax5.set_ylabel("Residual")
        ax5.set_xlabel("Time ($\\mu$s)")
        
        if show:
            plt.show()

        if path != '':
            plt.savefig(os.path.join(path, name))

        return fig


def generate_test_timetrace(a, tau, c, noise_sd, tEnd=2e-6, tInc=2.5e-10):
    '''Generates a timetrace of an exponential with a, tau, c
    and sprinkle in a normally distributed noise with a noise_sd.'''
    t = np.arange(0, tEnd, tInc)
    trace = a*np.exp(-t/tau) + c
    trace += np.random.normal(0, noise_sd, len(trace))

    return t, trace

def main():
    '''Test whether windowing works'''
    a, tau, c = [0.8, 1.2e-6, 0.]
    t, trace = generate_test_timetrace(a, tau, c, a/80)
    t2, trace2 = generate_test_timetrace(a, tau, c, a/80)

    ringdown = Ringdown(timetrace=trace, t=t, t0=0.4e-6, window=0.6e-6)
    ringdownb = Ringdown(timetrace=trace, t=t)
    print(ringdown.t0)
    print(ringdown.window)
    print(ringdownb.t0)
    print(ringdownb.window)

    ringdown.fit(additional_trace=trace2 - 0.01, plot=True, show=True)

if __name__ == '__main__':
    main()



