# https://bwengals.github.io/fitc-and-vfe.html
#

"""
Example
1 dimensional data, 50 data points
"""
import pymc3 as pm

import theano
import theano.tensor as tt
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(10)

nx, nu = 60, 7
x = np.random.rand(nx)
xu = np.random.rand(nu)
y = np.sin(2 * np.pi * x * 2.5) + 0.3 * np.random.randn(nx)

plt.plot(x, y, 'ko', ms=2);
plt.plot(xu, 0.0*np.ones(len(xu)), 'rx', ms=10)
x = x[:,None]; xu = xu[:,None]
xs = np.linspace(-0.1, 1.1, 100)

###############################################################

"""
Full GP
"""

with pm.Model() as model:
    ℓ = pm.Gamma("ℓ", alpha=1.5, beta=0.5)
    σ_f = pm.HalfCauchy("σ_f", beta=5)
    σ_n = pm.HalfCauchy("σ_n", beta=5)
    cov = tt.square(σ_f) * pm.gp.cov.ExpQuad(1, ℓ)
    gp = pm.gp.GP("gp", x, cov, sigma=σ_n, observed=y)
    trace = pm.sample(1000)

pm.traceplot(trace);

"""
Posterior sampling
"""
Xs = np.linspace(-0.2, 1.2, 150)[:, None]
with model:
    samples = pm.gp.sample_gp(trace, gp, Xs, n_samples=20, obs_noise=False)

plt.plot(x.flatten(), y, 'ko', ms=2);
plt.plot(Xs.flatten(), samples.T, "steelblue", alpha=0.4);
plt.title("Full GP");

###############################################################
"""
FITC
@3 arguments: inducing_points, n_inducing, approx
"""
with pm.Model() as model:
    ℓ = pm.Gamma("ℓ", alpha=1.5, beta=0.5)
    σ_f = pm.HalfCauchy("σ_f", beta=5)
    σ_n = pm.HalfCauchy("σ_n", beta=5)
    cov = tt.square(σ_f) * pm.gp.cov.Matern52(1, ℓ)
    gp = pm.gp.GP("gp", x, cov, sigma=σ_n, inducing_points=xu, observed=y)
    trace = pm.sample(1000)
pm.traceplot(trace);
"""
Posterior sampling
"""
with model:
    samples = pm.gp.sample_gp(trace, gp, Xs, n_samples=20, obs_noise=False)

plt.plot(x.flatten(), y, 'ko', ms=2);
plt.plot(Xs.flatten(), samples.T, "steelblue", alpha=0.4);
plt.title("FITC");
plt.plot(xu.flatten(), -2*np.ones(len(xu)), "rx");

###############################################################
"""
VFE with k-means
"""
with pm.Model() as model:
    ℓ = pm.Gamma("ℓ", alpha=1.5, beta=0.5)
    σ_f = pm.HalfCauchy("σ_f", beta=5)
    σ_n = pm.HalfCauchy("σ_n", beta=5)
    cov = tt.square(σ_f) * pm.gp.cov.Matern52(1, ℓ)
    gp = pm.gp.GP("gp", x, cov, sigma=σ_n, n_inducing=nu, approx="vfe", observed=y)
    trace = pm.sample(1000)
pm.traceplot(trace);
"""
Posterior sampling
"""
xs = np.linspace(-0.1, 1.1, 100)[:, None]
with model:
    samples = pm.gp.sample_gp(trace, gp, xs, n_samples=20, obs_noise=False)

plt.plot(x.flatten(), y, 'ko', ms=2);
xu = gp.distribution.Xu.flatten().eval()
plt.plot(xu.flatten(), -2 * np.ones(len(xu)), "rx", ms=10)
plt.plot(xs.flatten(), samples.T, "steelblue", alpha=0.2);
plt.title("VFE");
