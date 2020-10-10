import astropy.units as u
import numpy as np
from astropy.constants import c
import matplotlib.pyplot as plt

import naima
from naima.models import (
    ExponentialCutoffBrokenPowerLaw,
    InverseCompton,
    Synchrotron,
)


# Blob parameters
B = 1.e-3 * u.Gauss # magnetic field
R = 3.e16 * u.cm # radius
Gamma = 20. # bulk Lorentz factor
theta = 5*u.degree # viewing angle
delta = 1. / (Gamma * ( 1 - np.sqrt(1-Gamma**-2) * np.cos(theta) ))

# BLR parameters
T = 80 * u.K
w = 1 * u.eV * u.cm**-3

# electron distributions
# Model: ExponentialCutoffBrokenPowerLaw
# parameters for comoving frame
amplitude = 1.e35 / u.eV
e_0 = 1 * u.TeV # normalization energy
e_break = 10 * u.GeV
alpha_1 = 2.
alpha_2 = 3.
e_cutoff = 1 * u.TeV
beta = 1. # exponential cutoff rapidity 
Emin = 1 * u.GeV
Emax = 10 * u.TeV

comoving = ExponentialCutoffBrokenPowerLaw(
    amplitude = amplitude,
    e_0 = e_0,
    e_break = e_break,
    alpha_1 = alpha_1,
    alpha_2 = alpha_2,
    e_cutoff = e_cutoff,
    beta = beta,
)

lab = ExponentialCutoffBrokenPowerLaw(
    amplitude = amplitude * delta ** 3,
    e_0 = e_0 * delta,
    e_break = e_break,
    alpha_1 = alpha_1,
    alpha_2 = alpha_2,
    e_cutoff = e_cutoff * delta,
    beta = beta ,
)


# blob emission
#eopts = {"Eemax": Emax, "Eemin": Emin}

SYN = Synchrotron(comoving, B=B, Eemax=Emax, Eemin=Emin)

# Compute photon density spectrum from synchrotron emission 
Esy = np.logspace( np.log10(((Emin/u.TeV)**2*(B/u.Gauss)).cgs ) + 3 , np.log10(((e_cutoff/u.TeV)**2*(B/u.Gauss)).cgs) + 5, 300) * u.eV
Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
phn_sy = Lsy / (4 * np.pi * R ** 2 * c) * 2.24

SSC = InverseCompton(
    comoving,
    seed_photon_fields=[
        ["SSC", Esy, phn_sy],
    ],
    Eemax=Emax,
    Eemin=Emin,
)

EIC =  InverseCompton(
    lab,
    seed_photon_fields = [
        "CMB",
        ["BLR", T, w],
        ],
    Eemax = Emax * delta,
    Eemin = Emin * delta,
)

# Use matplotlib to plot the spectra
figure, ax = plt.subplots()

# Plot the computed model emission
energy = np.logspace(-3, 13, 100) * u.eV
SYN_plot = SYN.sed(energy / delta, distance = 0.) * delta**2
SSC_plot = SSC.sed(energy / delta, distance = 0.) * delta**2
EIC_plot = EIC.sed(energy, distance = 0.)
ylim_max = max(SYN_plot.max().value,SSC_plot.max().value,EIC_plot.max().value)
ax.set_ylim(ylim_max/1.e10, ylim_max*5)

for i, seed, ls in zip(
    range(2), ["CMB", "BLR"], ["--", "-."]
):
    ax.loglog(
        energy,
        EIC.sed(energy, distance=0., seed=seed),
        lw=2,
        c=naima.plot.color_cycle[i + 1],
        label=seed,
        ls=ls,
    )

ax.loglog(
    energy,
    SYN_plot,
    lw=2,
    c="g",
    label="SYN",
)
ax.loglog(
    energy,
    SSC_plot,
    lw=2,
    c="b",
    label="SSC",
)
ax.loglog(
    energy,
    EIC_plot,
    lw=2,
    c="r",
    label="EIC",
)

ax.legend(loc="lower right", frameon=False)
ax.set_xlabel("E, eV")
ax.set_ylabel("L, erg/s")
figure.tight_layout()
figure.savefig("SSC+EIC.png")
