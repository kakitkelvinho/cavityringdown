import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import pandas as pd

def exponential_decay(t, tau, A):
    return A*np.exp(-t/tau)

initial_guess = [1, 1e-6]

params, params_covariance = curve_fit(exponential_decay, )
