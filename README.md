# Ringdown 

## Introduction 

This is a package for analyzing the timetrace of a cavity ringdown experiment.

## Overview

A graphical summary or a cheatsheet of this package is as follows:

```python
# Structure

# Ringdowns
# ├Ringdown
# ├Ringdown
# ├Ringdown
# ├Ringdown
# ├Ringdown

# Working with csv
experiment = Experiment('path/to/folder')
# path to folder has structure /folder/setting/ringdown{n}.csv

# Methods with Ringdown
ringdown = Ringdown(timetrace=timetrace, t=t)
popt, pcov = ringdown.fit(plot=True)

# Methods with Ringdowns
ringdowns = Ringdowns(name="name", ringdowns=[ringdown1, ringdown2,...])
taus, tau_err = ringdowns.get_taus()
ringdowns.plot_taus()
```

## Usage

This package makes no assumptions about how to run your experiments and how you store your data. What it requires is an array of time and array of the intensity. By passing on these two arrays, in other words the $x$ and $y$ of your experiment, it can fit to it and tell you the decay time constants of experiment.

### Imports

The structure of the directory is as follows:
```python
analysis
├── loading.py
├── ringdown.py
└── ringdowns.py
```

To import the package to run in your script you can for example run:

```python
from ringdown.analysis.ringdowns import Ringdowns 
from ringdown.analysis.ringdown import Ringdown
```

Optionally, if your data structure permits (to be explained later):

```python
from ringdown.analysis.loading import Experiment
```

This offers a class to help you read into your experimental data and partition them into `Ringdown` and `Ringdowns`.

### Basic Usage

#### Loading in a Ringdown

The basic of classes are a `Ringdown` class, which is created by passing an array for time and an array for the timetrace.

```python
ringdown = Ringdown(timetrace=timetrace, t=t)
```
The numpy arrays `timetrace` and `t` could be obtained from your experimental data. For demonstration purposes, we can also generate a timetrace and 'load' it into an instance of the `Ringdown` class.
```python
# helper function in ringdown.py to generate a time trace
# used in the unit test

def generate_test_timetrace(a, tau, c, noise_sd, tEnd=2e-6, tInc=2.5e-10):
    '''Generates a timetrace of an exponential with a, tau, c
    and sprinkle in a normally distributed noise with a noise_sd.'''
    t = np.arange(0, tEnd, tInc)
    trace = a*np.exp(-t/tau) + c
    trace += np.random.normal(0, noise_sd, len(trace))

    return t, trace

# generate time array and timetrace
a, tau, c = [0.8, 1.2e-6, 0.]
t, trace = self.generate_timetrace(a, tau, c, a/80)

# instantiate a ringdown class
ringdown = Ringdown(timetrace=trace, t=t)
```

#### Fitting and Plotting

Upon instantiating the `Ringdown` class, the class would automatically set a window or a region of interest in which a fit can be performed. This is performed by identifying the start time of the decay, in which case is identified by the time at which the intensity has the maximum value. The window is set to a default value of 1.5 $\mu$s, but this can be adjusted manually either by directly accessing the attribute or by passing `window` as a parameter to the `fit()` method. 

Once the class is instantiated, a fit can be performed. This is simply performed by the method `fit()`, which is called as follows:
```python
ringdown = Ringdown(timetrace=timetrace, t=t)
popt, pcov = ringdown.fit()
```
We assume that the cavity ringdown is described by an exponential:
$$ I(t) = I_0 \exp(-\frac{t}{\tau}) + C $$
Programmatically, we denote $I_0$ by `a`, $\tau$ by `tau` and $C$ by `offset`. We also take the natural log of the decay to perform a (close to) linear fit using the `scipy.optimize.curve_fit` function. Thus, the fitting function, defined in the method `fit_func` is defined by `np.log(a*np.exp(-t/tau)+c)`, and calling the `fit` method performs a fit and returns the best fit parameters `popt` and the covariance matrix `pcov`. The order of the parameters are given as follows.
```python
popt, pcov = ringdown.fit(plot=True)
a, tau, c = popt
sigma_a, sigma_tau, sigma_c = np.sqrt(np.diag(pcov))
```
Passing `plot=True` as a parameter would also generate a graph with the following layout:
```python
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
```
In this figure:
1. Corresponds to a plot of the full timetrace with lines showing the window / region of interest (ROI)
2. Displays the log of the timetrace in the ROI and the fit over it
3. Displays the residuals of 2.
4. Displays the crop and normalized timetrace over the ROI and an intensity trace built by the best fit parameters, for visual inspection
5. Displays the residuals of 4.


#### Collection of Ringdowns

As with most experiments, you never only do one measurements. You often repeatedly take many such ringdowns with the same experimental parameters. For example, for our ringdown measurements, we keep all parameters the same except for the angle of  polarizer situated right before our detector. Therefore, we would repeat a ringdown measurement 50 times for having a polarizer at an oritentation of $50^\circ$. It would be inconvenient to instantiate a ringdown for each of the 50 measurements, say by running:
```python
# NOT RECOMMENDED but possible
ringdown1 = Ringdown(t=t0, timetrace=timetrace0)
ringdown1 = Ringdown(t=t1, timetrace=timetrace1)
ringdown2 = Ringdown(t=t2, timetrace=timetrace2)
ringdown3 = Ringdown(t=t3, timetrace=timetrace3)
ringdown4 = Ringdown(t=t4, timetrace=timetrace4)
# ... until 50, or ad infinitum
```
As such there is another class which allows you create a 'collection' of `Ringdown`. This class is known as `Ringdowns` and works like a python list, supporting in-place addition or appending with the `append` and `__add__` method. To instantiate `Ringdowns`, you pass in a name as an argument (which is also optional), and you can either incrementally add to its lists of `Ringdown` or instantiate it by passing the list to it already:

```python
import numpy as np

# lets say I generate a nested list of 50 time arrays and 50 timetraces
# define 50 tries, set parameters for my decay
n = 50
a = np.random.normal(0.7, 0.001, n)
taus = np.random.normal(1.9e-6, 1e-8, n)
c = 0.1

# instantiate my collection instance
ringdowns = Ringdowns(name='Polarizer 50deg')

for i in range(n):
    # generate t and timetrace
    t, trace = generate_test_timetrace(a[i], taus[i], c, noise_sd) 
    # store into ringdown
    ringdown = Ringdown(timetrace=trace, t=t)
    # place into ringdowns
    ringdowns += ringdown 
```

Upon instantiating a `Ringdowns` class, individual `Ringdown` can be accessed from this instance with list indexing. Each of the `Ringdown` can also have their methods be called.

#### Getting the distribution of fitted decay times

A reason why one would want to repeat measurements is to obtain an average and error for the fitted decay times over the many presumably identical measurements. This can be done by the following methods:
- `get_taus()` returns a tuple `(taus, tau_errs)`, in which the first element `taus` is a list of the fitted decay time and the second element is a list of the errors associated from each of the fitted decay time obtained from the covariance matrix (see section 7.2 of Hughes and Hase).
- `get_mean_taus()` and `get_std_taus()` calculates the sample mean and sample standard deviation of the decay times.
- `plot_taus()` outputs a plot of the decay times with their error bars. Helpful for visualization and locating where the mean is.


