import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit import cost
from scipy import stats
from IPython.core.display import Latex
from IPython.display import Math

# Measurement analysis functions
def error(data):
    return np.std(data)/np.sqrt(len(data))

def compiler(data):
    return np.mean(data), error(data)

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

def lin_func(x, a, b):
    return a*x+b

def gauss_pdf(x, mu, sigma):
    return 1.0 / np.sqrt(2*np.pi) / sigma * np.exp( -0.5 * (x-mu)**2 / sigma**2)



def lprint(*args,**kwargs):
    display(Latex('$$'+' '.join(args)+'$$'),**kwargs)

# Read in data
def read_data(filename):
    dat = np.genfromtxt(filename, delimiter='\t', names=('n', 't_s'))
    return dat

def period_fit(filename, sig_t, print_output=False, plot=False):
    data = read_data(filename)
    n, t = data['n'], data['t_s']
    # Build cost function
    cfit = cost.LeastSquares(n, t, sig_t, lin_func)

    # Run Minuit
    mfit = Minuit(cfit, a=9, b=0)
    mfit.migrad()

    residuals = t - lin_func(n, mfit.values['a'], mfit.values['b'])
    sig_t_new = np.std(residuals, ddof=1)
    
    if print_output:
        print(f"Updated uncertainty on time measurements: {sig_t_new:.3f} s")
    
    cfit = cost.LeastSquares(n, t, sig_t_new, lin_func)
    mfit = Minuit(cfit, a=9, b=0)
    mfit.migrad()
    
    # Print fit parameters with errors
    if print_output:
        for name in mfit.parameters:
            print(f"Fit value: {name} = {mfit.values[name]:.5f} ± {mfit.errors[name]:.5f}")
    
    # Chi2 info
    chi2_value = mfit.fval
    Ndof_value = len(t) - mfit.nfit
    prob = stats.chi2.sf(chi2_value, Ndof_value)
    if print_output:
        print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value}    Prob(Chi2,Ndof) = {prob:5.3f}")

    if plot:
        return n, t, mfit, residuals, sig_t_new, chi2_value, Ndof_value, prob
    else:
        return mfit.values['a'], mfit.errors['a'], sig_t_new
    
def plot_period_fit(filename, sig_t=0.1, bins=15):
    # --- Run fit (your function handles residual calc etc.) ---
    n, t, mfit, residuals, e_resid, chi2, Ndof, prob = period_fit(
        filename, sig_t, print_output=False, plot=True
    )

    # Extract fit parameters
    a, b = mfit.values["a"], mfit.values["b"]
    ea, eb = mfit.errors["a"], mfit.errors["b"]

    # Smooth line for fit curve
    xx = np.linspace(0, n[-1] + 1, 1000)
    yy = lin_func(xx, a, b)

    # ---------------- FIGURE LAYOUT ------------------
    fig, ax = plt.subplots(
        nrows=2, ncols=1,
        figsize=(16, 12),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True
    )

    # ---------------- MAIN PANEL ---------------------
    ax0 = ax[0]
    ax0.errorbar(n, t, yerr=sig_t, fmt="o", color="k", label="Time measurements")
    ax0.plot(xx, yy, "--", color="red", label="Fit")
    ax0.set_ylabel("Time elapsed (s)", fontsize=20)
    ax0.tick_params(labelsize=16)

    # Fit info block
    fit_info = [
        rf"$\chi^2/N_\mathrm{{dof}}$ = {chi2:.1f} / {Ndof}",
        rf"Prob = {prob:.3f}",
        rf"Period = ${a:.4f} \pm {ea:.4f}$ s",
        rf"Offset = ${b:.3f} \pm {eb:.3f}$ s"
    ]

    ax0.legend(title="\n".join(fit_info), fontsize=18, title_fontsize=18, alignment="left")
    ax0.text(
        0.01, 0.60,
        "Uncertainty from RMS of residuals",
        fontsize=20,
        transform=ax0.transAxes,
        va="top", ha="left"
    )

    # ---------------- RESIDUAL PANEL -----------------
    ax1 = ax[1]
    ax1.errorbar(n, residuals, yerr=e_resid, fmt="o", color="blue")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axhline(+e_resid, color="gray", linestyle="dotted")
    ax1.axhline(-e_resid, color="gray", linestyle="dotted")
    ax1.set_ylabel("Residuals (s)", fontsize=18)
    ax1.set_xlabel("Measurement number", fontsize=18)
    ax1.tick_params(labelsize=16)
    ax1.text(len(n)/2, -0.1, "Dashed lines show $\\pm 1\\sigma$", fontsize=16, fontweight="bold", ha="center")

    # ---------------- INSET: RESIDUAL HISTOGRAM ------------------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    xmin, xmax = -0.15, 0.15
    binwidth = (xmax - xmin) / bins
    x_gauss = np.linspace(xmin, xmax, 1000)

    counts, edges = np.histogram(residuals, bins=bins, range=(xmin, xmax))
    centers = (edges[:-1] + edges[1:]) / 2
    sy = np.sqrt(counts)

    valid = counts > 0

    ax_inset = ax0.inset_axes([0.60, 0.10, 0.35, 0.35])
    ax_inset.errorbar(
        centers[valid],
        counts[valid],
        yerr=sy[valid],
        xerr=binwidth / 2,
        fmt="none",
        ecolor="k",
        elinewidth=1
    )
    ax_inset.plot(
        x_gauss,
        gauss_pdf(x_gauss, 0, np.std(residuals)) * len(residuals) * binwidth,
        "r"
    )

    ax_inset.set_xlim(xmin, xmax)
    ax_inset.set_ylim(0, max(counts) + 2)
    ax_inset.set_xlabel("Residuals (s)", fontsize=14)
    ax_inset.set_ylabel("Frequency", fontsize=14)
    ax_inset.set_title("Distribution of time residuals", fontsize=16, fontweight="bold")
    ax_inset.tick_params(labelsize=14)

    return fig, ax

def read_csv(filename):
    """Read CSV from Waveforms"""
    dat = np.genfromtxt(filename, delimiter=',', skip_header=13, names=True)
    time = dat['Time_s']
    voltage = dat['Channel_1_V']
    return time, voltage

def find_midpoints(time, voltage, tol, return_all=False):
    """Find timing of ball crossings"""
    peaks_indx = np.argwhere((voltage > 2.2) & (voltage < 2.8))
    unique_indx = [peaks_indx[0]]
    for v in peaks_indx[1:]:
        if abs(v - unique_indx[-1]) > tol:
            unique_indx.append(v)
    time_peaks = time[unique_indx].flatten()
    volt_peaks = voltage[unique_indx].flatten()
    if return_all:
        return time_peaks, volt_peaks
    else:   
        return (time_peaks[1::2] + time_peaks[0::2]) / 2

def fit_func(x, a, b, c):
    return 0.5*a*x**2 + b*x + c

def fitter(xvals, yvals, yerrs):
    cfit = cost.LeastSquares(xvals, yvals, yerrs, fit_func)
    mfit = Minuit(cfit, a=0.14, b=0.7, c = 0.3)
    mfit.migrad();
    fit_a, _, _ = mfit.values[:]   # The fitted values of the parameters
    e_fit_a = mfit.errors['a']
    return fit_a, e_fit_a

def peak_plot(filename):
    time, voltage = read_csv(filename)
    timepass, voltpass = find_midpoints(time, voltage, tol=20, return_all=True)
    timepass_mid = (timepass[1::2] + timepass[0::2]) / 2
    voltpass_mid = (voltpass[1::2] + voltpass[0::2]) / 2
    timepass_mid_zero = timepass_mid - timepass_mid[0]
    timepass_zero = timepass-timepass_mid[0]
    time = time - timepass_mid[0]
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(time, voltage, 'ko', label='Signal data')
    ax.plot(timepass_zero, voltpass, 'rx', label='Detected Passings')
    ax.set_title("Data from Ball-on-Incline experiment", fontsize=20)
    ax.set_xlabel("Time since first peak (s)", fontsize=18)
    ax.set_ylabel("Voltage (V)", fontsize=18)
    ax.set_xlim(-0.4, 0.6)
    ax.tick_params(labelsize=14)
    # Inserted plot of first peak

    ax_inset = ax.inset_axes([0.08, 0.30, 0.25, 0.45])
    ax_inset.plot(time, voltage, 'k.', label='Signal data')
    ax_inset.plot(timepass_zero, voltpass, 'rx', label='Detected Passings')
    ax_inset.plot(timepass_mid_zero[0], voltpass_mid[0], 'r.', markersize=10, label='Midpoint of peak')
    ax_inset.set_title("Zoom: First detected peak", fontsize=14)
    ax_inset.set_xlabel("Time since first peak (s)", fontsize=12)
    ax_inset.set_ylabel("Voltage (V)", fontsize=12)
    ax_inset.set_xlim(-0.02, 0.02)
    ax_inset.tick_params(labelsize=12)
    ax_inset.legend(loc='lower center', fontsize=10);

    ax.legend();
    
def plot_gate_fit(filename, gate_pos, egate_pos):
    time, voltage = read_csv(filename)
    timepass = find_midpoints(time, voltage, tol=20)
    timepass = timepass - timepass[0]
    cfit = cost.LeastSquares(timepass, gate_pos, egate_pos, fit_func)
    mfit = Minuit(cfit, a=0.1, b=0.7, c = 0.3)
    mfit.migrad();
    fit_a, fit_b, fit_c = mfit.values[:]

    chi2_value = mfit.fval
    Ndof_value = len(gate_pos) - mfit.nfit
    Prob_value = stats.chi2.sf(chi2_value, Ndof_value)

    xx = np.linspace(timepass[0]-0.1, timepass[-1]+0.1, 1000)
    yy = fit_func(xx, fit_a, fit_b, fit_c)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(timepass, gate_pos, yerr=egate_pos, color='k', fmt='.', label='Measurements')
    ax.plot(xx, yy, '--', c='red', label='Fit')
    ax.set_title("Data from Ball-on-Incline experiment")
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Distance (m)", fontsize=18);
    fit_info = [f"$\\chi^2$ / $N_\\mathrm{{dof}}$ = {chi2_value:.1f} / {Ndof_value}", f"Prob($\\chi^2$, $N_\\mathrm{{dof}}$) = {Prob_value:.3f}",]
    for p, v, e in zip(mfit.parameters, mfit.values[:], mfit.errors[:]) :
        Ndecimals = max(0,-np.int32(np.log10(e)-1-np.log10(2)))
        fit_info.append(f"{p} = ${v:{10}.{Ndecimals}{"f"}} \\pm {e:{10}.{Ndecimals}{"f"}}$")
    ax.grid()
    ax.legend(title="\n".join(fit_info), fontsize=18, title_fontsize = 18, alignment = 'center');