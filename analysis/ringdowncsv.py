import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from analysis.loading import get_csv


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
    handfitdict: dict = field(init=False, repr=False)



    def __post_init__(self):
        # basic attributes
        self.name = self.csv_path.split('/')[-1].replace('.csv','')
        self.timetrace, self.tInc = get_csv(self.csv_path) 
        self.timetrace -= np.min(self.timetrace) + 1e-12 # 1e-12 to prevent log of zero
        self.n = len(self.timetrace)
        self.t = np.arange(0, self.n * self.tInc, self.tInc)
        self.auto_set_window()
        self.cropping_routine()

    def cropping_routine(self):
        self.croptime = self.t[self.crop_mask()]
        self.croptime_offset = self.croptime - self.croptime[0]
        self.croptimetrace = self.timetrace[self.crop_mask()]
        self.logtimetrace = np.log(self.croptimetrace)
        self.fit_by_hand()

    def auto_set_window(self, rolloff:float=0.5e-6, window_length:float=4e-6): 
        # cropping the basic attributes or taking the log
        self.t0 = self.t[np.argmax(self.timetrace[:round(self.n/2)])] + rolloff # find the max that lies in the first half of the array
        self.t1 = self.t0 + window_length

    def set_window(self, t0:float, t1:float):
        self.t0 = t0
        self.t1 = t1
        self.cropping_routine()

    def crop_mask(self):
        mask = np.logical_and(self.t >= self.t0, self.t <= self.t1)
        return mask

   
    # Fitting methods
    def fit_with_numpy(self):
        fit = np.polynomial.Polynomial.fit(self.croptime_offset, self.logtimetrace, deg=1)
        self.numpyfitobj = fit
        return fit

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
            'A': a,
            'B': b,
            'delta_y': delta_y,
            'delta_A': delta_a,
            'delta_B': delta_b,
        }
        return self.handfitdict

    def get_decay_constant(self, source="hand"):
        if source=="hand":
            b = self.handfitdict['B']
        else:
            b = self.numpyfitobj.convert().coef[1]
        return -1/b

    def estimate_finesse(self, source="hand", cavity_length=2e-2, verbose=False):
        # assuming a cavity length of 20cm
        tau = self.get_decay_constant(source)
        tau_err_frac = np.abs(self.handfitdict['delta_A']/self.handfitdict['A'])
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
        log_err = scope_err/self.croptimetrace
        var_est = np.sum(log_err*log_err)/self.n
        sd_est = np.sqrt(var_est)
        sd_fit = self.handfitdict["delta_y"]
        return sd_est, sd_fit


    # Plotting methods
    def plot_timetrace(self, save=False):
        plt.figure(figsize=(5,5));
        plt.plot(self.t, self.timetrace, label='timetrace');
        plt.axvline(self.t0, color='black');
        plt.axvline(self.t1, color='black');
        plt.title(f'{self.name} timetrace');
        plt.xlabel("time (s)")
        plt.ylabel("Voltage (V)")
        if save:
            plt.savefig('timetrace.png', dpi=300)
        plt.show();

    def plot_logtimetrace(self, save=False):
        fig = plt.figure(figsize=(10,10))
        plt.style.use('ggplot')
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        logplot =  fig.add_subplot(gs[0])
        residual = fig.add_subplot(gs[1])
        logplot.plot(self.croptime_offset, self.logtimetrace, label='log of timetrace',
                     marker='.', markersize=1, ls='');
        #logplot.errorbar(self.croptime_offset, self.logtimetrace, fmt='none', yerr=0.3, label="errors", ecolor="black", alpha=0.8)
        a = self.handfitdict['A']
        b = self.handfitdict['B']
        a_max, a_min = a+self.handfitdict['delta_A'], a-self.handfitdict['delta_A']
        b_max, b_min = b+self.handfitdict['delta_B'], b-self.handfitdict['delta_B']
        logplot.plot(self.croptime_offset, a+b*self.croptime_offset, label="fit", color='black')
        # we also plot given the errors the largest and smallest bounds for error
        logplot.plot(self.croptime_offset, a_max+b_min*self.croptime_offset, label="upper bound")
        logplot.plot(self.croptime_offset, a_min+b_max*self.croptime_offset, label="lower bound")
        logplot.set_title(f'{self.name} log timetrace');
        logplot.legend()
        logplot.set_xlabel("Time (s)")
        logplot.set_ylabel("log of voltage (v)")
        # we next plot the residuals
        residual.plot(self.croptime_offset, self.logtimetrace-a-b*self.croptime_offset,
                      marker='.', markersize=1, linestyle='')
        # we also plot the empirical/sample standard deviation
        # this is to check whether our points lie within 1 s.d. of our model
        residual.axhline(self.handfitdict['delta_y'], color='black', ls='--', label='+$\\sigma_y$')
        residual.axhline(-self.handfitdict['delta_y'], color='black', ls='--', label='-$\\sigma_y$')
        residual.set_ylabel("Residual")
        residual.set_xlabel("Time (s)")
        if save:
            plt.savefig('log_timetrace.png', dpi=300)
        plt.show();


def main():
    csv_path = '/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/PA_100/ringdown9.csv'
    ringdown = RingdownCSV(csv_path)
    prop_err = 7e-4/ringdown.croptimetrace
    var = np.sum(prop_err*prop_err)/len(ringdown.croptimetrace)
    print(f"var: {np.sqrt(var)}")
    print(f"sd_y: {ringdown.handfitdict['delta_y']}")
    print(f"finesse estimate: {ringdown.estimate_finesse()}")
    ringdown.plot_logtimetrace()
    


if __name__ == '__main__':
    main()
