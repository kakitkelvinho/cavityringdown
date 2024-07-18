import numpy as np
import csv
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class RingdownCSV:
    name: str =  field(init=False) # name of the ringdown
    csv_path: str # path to csv file (full path)
    timetrace: np.ndarray = field(init=False) # array to store timetrace
    logtimetrace: np.ndarray = field(init=False) # log of timetrace
    croptime: np.ndarray = field(init=False) # window out the time
    croptime_offset: np.ndarray = field(init=False) # shift array to start at t=0
    t: np.ndarray = field(init=False) # array to store time
    tInc: float = field(init=False) # increments of time
    t0: float = field(init=False) # start of window
    t1: float = field(init=False) # end of window
    n : float = field(init=False) # length of timetrace
    numpyfitobj: np.polynomial.Polynomial = field(init=False) # fit object
    handfitdict: dict = field(init=False)



    def __post_init__(self, rolloff:float=0.3e-6, window_length=5e-6):
        # basic attributes
        self.name = self.csv_path.split('/')[-1].replace('.csv','')
        self.timetrace, self.tInc = get_csv(self.csv_path) 
        self.timetrace -= np.min(self.timetrace) + 1e-12 # 1e-12 to prevent log of zero
        self.n = len(self.timetrace)
        self.t = np.arange(0, self.n * self.tInc, self.tInc)
        self.t0 = self.t[np.argmax(self.timetrace)] + rolloff
        self.t1 = self.t0 + window_length
        # cropping the basic attributes or taking the log
        self.croptime = self.t[self.crop_mask()]
        self.croptime_offset = self.croptime - self.croptime[0]
        self.logtimetrace = np.log(self.crop_timetrace())
        self.fit_by_hand()
        
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

    def estimate_finesse(self, source="hand", cavity_length=2e-2):
        # assuming a cavity length of 20cm
        tau = self.get_decay_constant(source)
        return np.pi*3e8*tau/cavity_length


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
        plt.figure(figsize=(10,10))
        plt.style.use('ggplot')
        plt.plot(self.croptime_offset, self.logtimetrace, label='log of timetrace');
        #if self.numpyfitobj:
        #    plt.plot(self.croptime_offset, self.numpyfitobj(self.croptime_offset), label="numpy fit")
        a = self.handfitdict['A']
        b = self.handfitdict['B']
        plt.plot(self.croptime_offset, a+b*self.croptime_offset, label="fit")
        plt.title(f'{self.name} log timetrace');
        plt.legend()
        plt.xlabel("time (s)")
        plt.ylabel("Log of Voltage (V)")
        if save:
            plt.savefig('log_timetrace.png', dpi=300)
        plt.show();

  

    def set_window(self, t0:float, t1:float):
        self.t0 = t0
        self.t1 = t1

    def crop_mask(self):
        mask = np.logical_and(self.t >= self.t0, self.t <= self.t1)
        return mask

    def crop_timetrace(self):
        self.timetrace = self.timetrace[self.crop_mask()]
        return self.timetrace




def get_csv(filename: str, index:int=0):
    timetrace = []
    with open(filename, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        tInc = 0.
        tmp = 0.
        for row in csv_reader:
            if line_count == 0:
                headers = row
                tInc = [header for header in headers if "tInc" in header]
                tInc = float(tInc[0].split('=')[-1].split('s')[0])
                line_count += 1
            else:
                channel = row[index]
                try:
                    float(channel)
                except Exception as e:
                    print(f"{e} with row {line_count}, replace {channel} with {tmp}")
                    channel = tmp
                timetrace.append(float(channel))
                tmp = channel
                line_count += 1
    timetrace = np.array(timetrace)
    return timetrace, tInc
 



def main():
    csv_path = '/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/PA_100/ringdown9.csv'
    ringdown = RingdownCSV(csv_path)
    print(f"finesse estimate: {ringdown.estimate_finesse()}")
    ringdown.plot_logtimetrace()
    


if __name__ == '__main__':
    main()
