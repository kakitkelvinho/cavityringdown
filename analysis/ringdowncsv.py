import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from analysis.loading import get_csv
from scipy.optimize import curve_fit
from uncertainties import ufloat


@dataclass
class RingdownCSV:
    name: str =  field(default="None provided", init=False) # name of the ringdown
    csv_path: str = field(repr=False) # path to csv file (full path)
    timetrace: np.ndarray = field(init=False, repr=False) # array to store timetrace
    croptimetrace: np.ndarray = field(init=False, repr=False) # crop of timetrace
    logtimetrace: np.ndarray = field(init=False, repr=False) # log of timetrace (from crop of timetrace)
    croptime: np.ndarray = field(init=False, repr=False) # window out the time
    croptime_offset: np.ndarray = field(init=False, repr=False) # shift array to start at t=0
    t: np.ndarray = field(init=False, repr=False) # array to store time
    tInc: float = field(init=False, repr=False) # increments of time
    t0: float = field(init=False, repr=False) # start of window
    t1: float = field(init=False, repr=False) # end of window
    n : float = field(init=False, repr=False) # length of timetrace
    numpyfitobj: np.polynomial.Polynomial = field(init=False, repr=False) # fit object
    fitdict: dict = field(init=False, repr=False)
    scope_error: float = field(default=8.34e-4, repr=False)



    def __post_init__(self):
        # basic attributes
        self.name = self.csv_path.split('/')[-1].replace('.csv','')
        self.timetrace, self.tInc = get_csv(self.csv_path) 
        self.timetrace -= np.min(self.timetrace) 
        #self.timetrace -= 0.003417 # 1e-12 to prevent log of zero
        self.n = len(self.timetrace)
        self.t = np.arange(0, self.n * self.tInc, self.tInc)
        self.auto_set_window()
        self.cropping_routine()

    def test_timetrace(self, timetrace: np.ndarray, t: np.ndarray, noise: float):
        self.croptimetrace = timetrace
        self.croptime_offset = t
        self.logtimetrace = np.log(self.croptimetrace)
        self.scope_error = noise
        self.fit_with_scipy()

    def cropping_routine(self):
        self.croptime = self.t[self.crop_mask()]
        self.croptime_offset = self.croptime - self.croptime[0]
        latermask = self.crop_mask(self.t0+2e-6, self.t0+3e-6)
        self.croptimetrace = self.timetrace[self.crop_mask()]
        noise = self.timetrace[latermask]
        noise_level = np.median(noise)
        self.logtimetrace = np.log(self.croptimetrace/np.max(self.croptimetrace))
        #self.logtimetrace = np.log(self.croptimetrace)
        self.fit_with_scipy()

    def auto_set_window(self, rolloff:float=1.1e-6, window_length:float=3e-6): 
        # cropping the basic attributes or taking the log
        self.t0 = self.t[np.argmax(self.timetrace[:round(self.n/2)])] + rolloff # find the max that lies in the first half of the array
        self.t1 = self.t0 + window_length

    def set_window(self, t0:float, t1:float):
        self.t0 = t0
        self.t1 = t1
        self.cropping_routine()

    def crop_mask(self, t0=None, t1=None):
        if t0 == None:
            t0 = self.t0
        if t1 == None:
            t1 = self.t1
        mask = np.logical_and(self.t >= t0, self.t <= t1)
        return mask

   
    # Fitting methods
    def fit_with_numpy(self):
        fit = np.polynomial.Polynomial.fit(self.croptime_offset, self.logtimetrace, deg=1)
        self.numpyfitobj = fit
        return fit


    def fit_log_offset(self, plot=True, test=False):
        t = self.croptime_offset
        if test:
            a_orig, tau_orig, offset_orig = [1, 2.5e-6, 0.5]
            trace = a_orig*np.exp(-(1/tau_orig)*t)+offset_orig
            noise_sd = 0.09
            trace += np.random.normal(0, noise_sd, len(trace))
        else:
            trace = self.croptimetrace 
            max_croptimetrace = np.max(self.croptimetrace)
            trace /= max_croptimetrace
            noise_sd = self.scope_error

        trace = np.log(trace)
        f = lambda t, tau, a, offset: np.log(a*np.exp(-t/tau)+offset)        
        popt, pcov = curve_fit(f, t, trace, p0=[2e-6, 0.08, 0])
        tau, a, offset = popt
        delta_tau, delta_a, delta_offset = np.sqrt(np.diag(pcov))
        residual = trace - f(t, *popt)
        mse = np.sum(residual**2)/(len(trace)-3)
        prop_err_sq = (noise_sd/np.exp(trace))**2 + (noise_sd/max_croptimetrace)**2
        var_est = np.mean(prop_err_sq)
        print(f"MSE: {mse}, Estimator of var: {var_est}")
        print(f"parameters: tau: {tau}, a: {a}, offset: {offset}")
        print(f"error: tau: {delta_tau}, a: {delta_a}, offset: {delta_offset}")
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(10,10))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1,])
            ax1 = fig.add_subplot(gs[0]) 
            ax2 = fig.add_subplot(gs[1])
            ax1.plot(t, trace, label="time trace", marker=1, ls='', color='red')
            ax1.plot(t, f(t, *popt), label="fit")
            ax1.set_xlabel("t (s)")
            ax1.set_ylabel("Voltage (V)")
            ax1.legend()
            ax2.plot(t, residual, ls='', label="residual")
            ax2.axhline(noise_sd, color='gray', ls='--')
            ax2.axhline(-noise_sd, color='gray', ls='--')
            ax2.set_ylabel('Residual')
            ax2.set_xlabel('t (s)')
            ax2.legend()
            fig.tight_layout()
            plt.savefig('./results/test.png') 
            plt.show()




    def fit_exponential_curve(self, plot=True, test=False):
        t = self.croptime_offset
        if test:
            a_orig, tau_orig, offset_orig = [0.09, 2e-6, -0.001]
            trace = a_orig*np.exp(-(1/tau_orig)*t)+offset_orig
            noise_sd = 0
            trace += np.random.normal(0, noise_sd, len(trace))
        else:
            trace = self.croptimetrace
            noise_sd = self.scope_error
        f = lambda t, tau, a, offset: a*np.exp(-t/tau) + offset
        popt, pcov = curve_fit(f, t, trace, p0=[2e-6, 0.08, 0])
        tau, a, offset = popt
        delta_tau, delta_a, delta_offset = np.sqrt(np.diag(pcov))
        residual = trace - f(t, *popt)
        mse = np.sum(residual**2)/(len(trace)-3)
        print(f"MSE: {mse}")
        print(f"parameters: tau: {tau}, a: {a}, offset: {offset}")
        print(f"error: tau: {delta_tau}, a: {delta_a}, offset: {delta_offset}")
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(10,10))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1,])
            ax1 = fig.add_subplot(gs[0]) 
            ax2 = fig.add_subplot(gs[1])
            ax1.plot(t, trace, label="time trace", marker=1, ls='', color='red')
            ax1.plot(t, f(t, *popt), label="fit")
            ax1.set_xlabel("t (s)")
            ax1.set_ylabel("Voltage (V)")
            
            ax2.plot(t, residual, ls='', color='green', label="residual")
            #ax2.axhline(noise_sd**2,color='gray', ls='--', label="noise var")
            ax2.axhline(noise_sd, color='gray', ls='--')
            ax2.axhline(-noise_sd, color='gray', ls='--')
            ax2.set_ylabel('Residual')
            ax2.set_xlabel('t (s)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('./results/test.png') 

    def fit_with_scipy(self):
        f = lambda x, m, c: m*x + c
        popt, pcov = curve_fit(f, self.croptime_offset, self.logtimetrace)
        m, c = popt[0], popt[1]
        delta_m, delta_c = np.sqrt(np.diag(pcov))
        residual = self.logtimetrace - f(self.croptime_offset, *popt)
        delta_y = np.sqrt(np.sum(residual**2)/(len(self.croptime_offset)-2))
        self.fitdict = {
            'm': m,
            'c': c,
            'delta_m': delta_m,
            'delta_c': delta_c,
            'delta_y': delta_y
        }
        return self.fitdict
    
    def fit_by_hand(self):
        """Implement fitting by normal equations,
        where the model is y = A + B x"""
        sum_x = np.sum(self.croptime_offset)
        sum_xsq = np.sum(self.croptime_offset*self.croptime_offset)
        sum_xy = np.sum(self.croptime_offset*self.logtimetrace)
        sum_y = np.sum(self.logtimetrace)
        n = len(self.logtimetrace)
        denominator = n * sum_xsq - sum_x**2

        # implement the normal equations to find A and B
        a = (sum_xsq*sum_y - sum_x*sum_xy) / denominator
        b = (n * sum_xy - sum_x*sum_y) / denominator

        # estimate standard deviations
        spread = self.logtimetrace-a-b*self.croptime_offset
        delta_y = np.sqrt((1/(n-2))*np.sum(spread*spread))
        delta_a = delta_y * np.sqrt(sum_xsq/denominator)
        delta_b = delta_y * np.sqrt(n/denominator)
        self.handfitdict = {
            'c': a,
            'm': b,
            'delta_y': delta_y,
            'delta_c': delta_a,
            'delta_m': delta_b,
        }
        return self.fitdict

    def get_decay_constant(self):
        m = self.fitdict['m']
        m = ufloat(self.fitdict['m'], self.fitdict['delta_m'])
        return -1/m

    def estimate_finesse(self, cavity_length=2e-2, verbose=False):
        # assuming a cavity length of 20cm
        tau = self.get_decay_constant()
        tau_err_frac = tau.s/tau.n # float; fractional uncertainty
        l_err_frac = 500e-6/cavity_length # worse case in which the cavity length differs by one FSR
        sum_in_quad = lambda x: np.sqrt(np.sum(x*x))
        total_frac = sum_in_quad(np.array([tau_err_frac, l_err_frac]))
        finesse =  np.pi*299792458*tau/cavity_length
        error = total_frac*finesse
        if verbose:
            print(f"tau fractional uncertainty: {tau_err_frac}")
            print(f"length fractional uncertainty: {l_err_frac}")
            print(f"Total fractional uncertainty: {total_frac}")
            print(f"Finesse: {finesse}({error})")
        return finesse, error

    def compare_errors(self, scope_err=7e-4):
        """Compare the empirical error (propagated) wit the fit error."""
        log_err = ((scope_err/self.croptimetrace)**2 + (scope_err/np.max(self.croptimetrace))**2)
        var_est = np.sum(log_err*log_err)/len(self.croptimetrace)
        sd_est = np.sqrt(var_est)
        sd_fit = self.fitdict["delta_y"]
        return sd_est, sd_fit


    # Plotting methods
    def plot_timetrace(self, save=False):
        plt.figure(figsize=(5,5));
        plt.plot(self.t, self.timetrace, label='timetrace');
        plt.axvline(self.t0, color='black');
        plt.axvline(self.t1, color='black');
        plt.title(f'{self.name} timetrace');
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        if save:
            plt.savefig('timetrace.png', dpi=300)
        plt.show();
        plt.close();

    def plot_logtimetrace(self, save=False, show=True, figname:str = "log_timetrace"):
        fig = plt.figure(figsize=(10,10))
        plt.style.use('ggplot')
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1,])
        logplot =  fig.add_subplot(gs[0])
        residual = fig.add_subplot(gs[1])
        #error = fig.add_subplot(gs[2])
        logplot.plot(self.croptime_offset, self.logtimetrace, label='log of timetrace',
                     marker='.', markersize=1, ls='');
        #logplot.errorbar(self.croptime_offset, self.logtimetrace, fmt='none', yerr=0.3, label="errors", ecolor="black", alpha=0.8)
        m = self.fitdict['m']
        c = self.fitdict['c']
        m_max, m_min = m + self.fitdict['delta_m'], m - self.fitdict['delta_m']
        c_max, c_min = c + self.fitdict['delta_c'], c - self.fitdict['delta_c']
        logplot.plot(self.croptime_offset, c+m*self.croptime_offset, label=f"fit {-1/m:0.2e}", color='black')
        # we also plot given the errors the largest and smallest bounds for error
        logplot.plot(self.croptime_offset, c_max+m_min*self.croptime_offset, label="upper bound")
        logplot.plot(self.croptime_offset, c_min+m_max*self.croptime_offset, label="lower bound")
        logplot.set_title(f'{self.name} log timetrace');
        logplot.legend()
        logplot.set_xlabel("Time (s)")
        logplot.set_ylabel("log of voltage (v)")
        # we next plot the residuals
        residuals = self.logtimetrace-c-m*self.croptime_offset
        residual.plot(self.croptime_offset, residuals**2,
                      marker='.', markersize=1, linestyle='', color='tab:orange', label='residual squared')
        residual.plot(self.croptime_offset, (self.scope_error/self.croptimetrace)**2 + (self.scope_error/np.max(self.croptimetrace))**2, color='tab:green', marker='.', markersize=1, ls='', label='err. var')
        # we also plot the empirical/sample standard deviation
        # this is to check whether our points lie within 1 s.d. of our model
        residual.axhline(self.fitdict['delta_y']**2, color='black', ls='--', label='$+\\sigma_y^2$')
        #residual.axhline(-self.fitdict['delta_y'], color='black', ls='--', label='$-\\sigma_y$')
        residual.set_ylabel("Residual")
        residual.set_xlabel("Time (s)")
        residual.legend()
        # error plot
        #error.plot(self.croptime_offset, np.sqrt(self.scope_error/self.croptimetrace)**2 + (self.scope_error/np.max(self.croptimetrace))**2)
        if save:
            plt.savefig(f'./results/log_timetrace/{figname}.png', dpi=300)
        if show:
            plt.show();
        # generate an array for propagated uncertainty

        plt.close()



def main():
    csv_path = '/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/PA_100/ringdown9.csv'
    ringdown = RingdownCSV(csv_path)
    prop_err = 7e-4/ringdown.croptimetrace
    var = np.sum(prop_err*prop_err)/len(ringdown.croptimetrace)
    print(f"var: {np.sqrt(var)}")
    print(f"sd_y: {ringdown.fitdict['delta_y']}")
    print(f"finesse estimate: {ringdown.estimate_finesse()}")
    ringdown.plot_logtimetrace()
    


if __name__ == '__main__':
    main()
