from collections import Counter

import GPy
import numpy as np
from jax import jit
import jax.numpy as jnp
from scipy.special import gamma
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from tqdm import tqdm


def gp_covariance_matrix(side_length=16, var=1, scale=1.7):
    xx, yy = jnp.mgrid[0:side_length, 0:side_length]
    X = jnp.vstack((xx.flatten(), yy.flatten())).T
    k = GPy.kern.RBF(input_dim=2, variance=var, lengthscale=scale)  # define kernel
    return k.K(X)  # compute covariance matrix


def aggregated_covariance_matrix(K, side_length, patch_length):
    # K is the covariance matrix for the multivariate gaussian
    # that the full value function is sampled from. It is of shape
    # (side_length**2, side_length**2). We want to create a new
    # matrix that would correspond to the covariance matrix of
    # a multivariate gaussian that produces aggregated value functions

    if patch_length == 1:
        return K

    # first, reshape K to be 4d where K[i, j, k, l] is the covariance
    # between the value at position (i, j) and the value at position (k, l)
    K = K.reshape((side_length, side_length, side_length, side_length))

    # initialise new covariance matrix
    new_length = side_length // patch_length
    new_K = jnp.zeros((new_length, new_length, new_length, new_length))

    # loop over all patches
    pbar = tqdm(total=int(new_length**4))
    pbar.set_description(
        f"computing aggregated covariance matrix (patch length={patch_length})"
    )
    for i in range(new_length):
        for j in range(new_length):
            for k in range(new_length):
                for l in range(new_length):
                    # for every point in patch (i,j), compute the covariances with
                    # every point in patch (k,l), sum them and divide by patch_length**2
                    covs = K[
                        i * patch_length : (i + 1) * patch_length,
                        j * patch_length : (j + 1) * patch_length,
                        k * patch_length : (k + 1) * patch_length,
                        l * patch_length : (l + 1) * patch_length,
                    ]
                    new_K = new_K.at[i, j, k, l].set(jnp.sum(covs) / patch_length**2)
                    pbar.update(1)

    # now reshape new_K to be 2d
    return new_K.reshape((new_length**2, new_length**2))


def gaussian_entropy(cov):
    n = cov.shape[0]
    nats = jnp.log(jnp.linalg.det(cov)) + (n * jnp.log(2 * jnp.pi * jnp.e))
    nats /= 2
    return nats / jnp.log(2)  # convert to bits


def uniform_entropy(n):
    return jnp.log(n) / jnp.log(2)


def clipped_gaussian(mus, stds, num=1):
    cov = np.diag(stds**2)
    samples = np.random.default_rng().multivariate_normal(mus, cov, num)
    return np.clip(samples, 0, 1)


def boltzmann1d(x, beta):
    p = jnp.exp(x / beta)
    return p / jnp.sum(p)


def boltzmann2d(x, beta):
    p = jnp.exp(x / beta)
    return p / jnp.sum(p, axis=1).reshape((-1, 1))


def crp(z, c, log=True):
    counts = Counter(z)
    coeff = ((c ** len(counts)) * gamma(c)) / gamma(c + len(z))

    if log:
        return coeff + jnp.sum([gamma(counts[k]) for k in counts])

    return coeff * jnp.prod([gamma(counts[k]) for k in counts])


def dirichlet_multinomial(counts, T, L, g, log=True):
    coeff = (gamma(L * g) * gamma(T + 1)) / gamma(T + (L * g))
    prods = jnp.prod(gamma(counts + g) / (gamma(g) * gamma(counts + 1)), axis=1)

    tmp = coeff * prods
    return jnp.sum(jnp.log(tmp)) if log else jnp.prod(tmp)


def multivariate_gaussian_model(data):
    dim = data.shape[-1]
    mu = numpyro.sample("mu", dist.Normal(0.5 * jnp.ones(dim), jnp.ones(dim)))
    sigma = numpyro.sample(
        "sigma", dist.InverseGamma(0.5 * jnp.ones(dim), 0.5 * jnp.ones(dim))
    )
    cov = jnp.diag(sigma)
    numpyro.sample("obs", dist.MultivariateNormal(mu, cov), obs=data)


def posterior_predictive(rng_key, model, data, num_warmpup=500, num_samples=1000):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmpup, num_samples=num_samples)
    mcmc.run(rng_key, data)
    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples, num_samples=1, batch_ndims=0)
    return predictive(rng_key, data)
