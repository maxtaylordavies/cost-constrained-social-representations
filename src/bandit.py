from collections import Counter

import jax
from jax import random
import jax.numpy as jnp
from tqdm import tqdm

from src.probabilistic import boltzmann1d


def sample_value_functions_ind(rng_key, K, side_length, num_agents):
    Vs = jnp.zeros((num_agents, side_length, side_length))
    for i in range(num_agents):
        v = random.multivariate_normal(rng_key + i, jnp.zeros(side_length**2), K)
        v = ((v - v.min()) / (v.max() - v.min())).reshape((side_length, side_length))
        Vs = Vs.at[i].set(v)

    return Vs


def sample_value_functions_grp(key, K, side_length, num_groups, agents_per_group, rho):
    # sample central value function for each group
    group_Vs = jnp.zeros((num_groups, side_length, side_length))
    for i in range(num_groups):
        v = random.multivariate_normal(key + i, jnp.zeros(side_length**2), K)
        v = ((v - v.min()) / (v.max() - v.min())).reshape((side_length, side_length))
        group_Vs = group_Vs.at[i].set(v)

    # set group assignments
    zs = jnp.repeat(jnp.arange(num_groups), agents_per_group)
    zs = zs.reshape((num_groups, agents_per_group)).T.flatten()

    # sample individual value functions based on group assignments
    Vs = jnp.zeros((num_groups * agents_per_group, side_length, side_length))
    for i, z in enumerate(zs):
        v = random.multivariate_normal(
            key + num_groups + i, group_Vs[z].reshape((-1,)), rho * K
        )
        v = ((v - v.min()) / (v.max() - v.min())).reshape((side_length, side_length))
        Vs = Vs.at[i].set(v)

    return group_Vs, zs, Vs


def compute_distance_matrices(side_length):
    dists = jnp.zeros((side_length, side_length, side_length, side_length), dtype=int)

    xx, yy = jnp.mgrid[0:side_length, 0:side_length]
    X = jnp.vstack((xx.flatten(), yy.flatten())).T.reshape((side_length, side_length, 2))

    for i in range(side_length):
        for j in range(side_length):
            start = jnp.array([i, j]).reshape((1, 1, 2))
            d = jnp.sum(jnp.abs(X - start), axis=2).reshape((side_length, side_length))
            dists = dists.at[i, j].set(d)

    return dists


def simulate_bandit_choices(rng_key, V, beta, N, positions=None, dists=None, c=0):
    side_length = V.shape[0]
    num_arms = side_length**2

    if positions is None or c == 0:
        p = boltzmann1d(V, beta).reshape((-1,))
        return random.choice(rng_key, num_arms, shape=(N,), p=p)

    # precompute probability vectors for all possible start positions
    P = jnp.exp((V - c * dists) / beta).reshape((side_length, side_length, num_arms))

    # create vectorized function to execute over provided positions
    def f(p):
        return random.choice(rng_key, num_arms, shape=(1,), p=p)

    # sample choices
    probs = P[positions[:, 0], positions[:, 1]]
    return jax.vmap(f)(probs).reshape((-1,))


def imitation_return(choices, Vself, positions=None, dists=None, c=0):
    Vself = Vself.reshape((-1,))
    if positions is None or c == 0:
        return jnp.mean(Vself[choices])

    # combine last two dimensions of dists
    l = dists.shape[0]
    dists = dists.reshape((l, l, l**2))

    returns = jnp.zeros(len(positions))
    for i, choice in enumerate(choices):
        r = Vself[choice] - (c * dists[positions[i][0], positions[i][1], choice])
        returns = returns.at[i].set(r)

    return jnp.mean(returns)


def v_similarity(V1, V2):
    return jnp.dot(V1, V2) / (jnp.linalg.norm(V1) * jnp.linalg.norm(V2))


def compute_weights_val(chi_self, chi):
    weights = jnp.zeros(len(chi))
    for i, x in enumerate(chi):
        w = v_similarity(chi_self[0], x[0])
        weights = weights.at[i].set(w)
    return weights


def compute_weights_val_beta(chi_self, chi, gamma=0.5):
    weights = jnp.zeros(len(chi))
    for i, x in enumerate(chi):
        w = v_similarity(chi_self[0], x[0])
        w -= gamma * jnp.log10(x[1])
        weights = weights.at[i].set(w)
    return weights


def compute_weights_uniform(chi_self, chi):
    return jnp.ones(len(chi)) / len(chi)


def choose_agents(rng_key, w_func, beta_self, chi_self, chi, num_trials=1):
    w = w_func(chi_self, chi)
    p = boltzmann1d(w, beta_self)
    return random.choice(rng_key, len(chi), p=p, shape=(num_trials,), replace=True)


def simulate_imitation(
    rng_key,
    weight_algo,
    Vself,
    beta_self,
    Vs,
    betas,
    chi_self=None,
    chi=None,
    num_trials=1000,
):
    if chi is None:
        chi = [(Vs[i], betas[i]) for i in range(len(Vs))]

    if chi_self is None:
        chi_self = (Vself, beta_self)

    if weight_algo == "indiscriminate":
        w_func = compute_weights_uniform
    elif weight_algo == "val":
        w_func = compute_weights_val
    elif weight_algo == "val_beta":
        w_func = compute_weights_val_beta
    else:
        raise ValueError(f"Unrecognised weighting algorithm: {weight_algo}")

    # simulate bandit choices
    choice_matrix = []
    for i, V in enumerate(Vs):
        key = random.PRNGKey(i)
        choice_matrix.append(simulate_bandit_choices(key, V, betas[i], num_trials))
    choice_matrix = jnp.stack(choice_matrix, axis=1)

    # simulate selective imitation
    agent_idxs = choose_agents(rng_key, w_func, beta_self, chi_self, chi, num_trials)
    imitated_choices = choice_matrix[jnp.arange(num_trials), agent_idxs]
    return imitation_return(imitated_choices, Vself)
