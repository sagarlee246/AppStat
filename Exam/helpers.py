import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit import cost
from scipy import stats
from IPython.core.display import Latex
from IPython.display import Math

# Measurement analysis functions
def error_esti(data):
    return np.std(data, ddof=1)/np.sqrt(len(data))

def error_true(data):
    return np.std(data)/np.sqrt(len(data))

def compiler_esti(data):
    return np.mean(data), error_esti(data)

def compiler_true(data):
    return np.mean(data), error_true(data)

def weighted_mean(data, errors):
    mean = np.average(data, weights=1/(errors**2))
    error = np.sqrt(1/np.sum(1/(errors**2)))
    return mean, error

def chi2_and_prob(data, model, errors, n_fixed_params, print_output=False):
    chi2 = np.sum((data - model)**2 / (errors**2))
    prob = stats.chi2.sf(chi2, len(data)-n_fixed_params)
    if print_output:
        Ndof = len(data)-n_fixed_params
        print(f"Chi2 value: {chi2:.2f}   Ndof = {Ndof}    Prob(Chi2,Ndof) = {prob:5.4f}")
    else:
        return chi2, prob
def lprint(*args,**kwargs):
    display(Latex('$$'+' '.join(args)+'$$'),**kwargs)