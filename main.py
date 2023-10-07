import argparse
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import jax
from jax import random
import jax.numpy as jnp
from tqdm import tqdm
import seaborn as sns

from src.probabilistic import (
    gp_covariance_matrix,
    aggregated_covariance_matrix,
    gaussian_entropy,
    uniform_entropy,
    boltzmann1d,
)
from src.bandit import (
    sample_value_functions_ind,
    sample_value_functions_grp,
    simulate_bandit_choices,
    compute_distance_matrices,
    imitation_return,
    v_similarity,
)
from src.utils import mean_pool_1d, mean_pool_2d

jax.config.update("jax_enable_x64", True)


def plot_value_functions(Vs, title=None):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.flatten()

    for i, af in enumerate([0, 2, 4, 6]):
        Vhats = [mean_pool_1d(V, 2**af) for V in Vs]
        title = (
            "example true value functions" if af == 0 else f"compressed (factor = {2 ** af})"
        )
        axs[i].imshow(jnp.stack(Vhats, axis=0))
        axs[i].set_title(title, fontsize=12)
        axs[i].axis("off")

    fig.tight_layout()
    fig.savefig("results/bandit/example-value-functions.svg")


def get_bandit_choices(key, Vs, side_length, beta, num_trials, c):
    positions = random.randint(key, (num_trials, 2), 0, side_length)
    dists = compute_distance_matrices(side_length)

    def f(v):
        return simulate_bandit_choices(key, v, beta, num_trials, positions, dists, c)

    choices = jax.vmap(f)(Vs)
    return choices, positions, dists


def indiscriminate_policy(
    key, choices, Vself, Vs, beta_self, af, positions=None, dists=None, c=0, stochastic=False
):
    weights = jnp.ones(len(Vs)) / len(Vs)
    return compute_imitation_return(
        key,
        choices,
        weights,
        Vself,
        beta_self,
        positions=positions,
        dists=dists,
        c=c,
        stochastic=True,
    )


def individual_agents_policy(
    key, choices, Vself, Vs, beta_self, af, positions=None, dists=None, c=0, stochastic=False
):
    Vself_hat = mean_pool_2d(Vself, 2**af).reshape((-1,))
    V_hats = [mean_pool_2d(V, 2**af).reshape((-1,)) for V in Vs]

    weights = jnp.array([v_similarity(Vself_hat, Vhat) for Vhat in V_hats])
    weights /= weights.max()

    return compute_imitation_return(
        key,
        choices,
        weights,
        Vself,
        beta_self,
        positions=positions,
        dists=dists,
        c=c,
        stochastic=stochastic,
    )


def group_agents_policy(
    key,
    choices,
    Vself,
    group_Vs,
    z,
    beta_self,
    af,
    positions=None,
    dists=None,
    c=0,
    stochastic=False,
):
    # Vself_hat = mean_pool_2d(Vself, 2**af).reshape((-1,))
    # group_V_hats = jnp.stack([mean_pool_2d(V, 2**af).reshape((-1,)) for V in group_Vs])
    # V_hats = group_V_hats[z]

    # weights = jnp.array([v_similarity(Vself_hat, V) for V in V_hats])
    # weights /= weights.max()

    Vs = group_Vs[z]
    weights = jnp.array([v_similarity(Vself.reshape((-1,)), V.reshape((-1,))) for V in Vs])
    weights /= weights.max()

    return compute_imitation_return(
        key,
        choices,
        weights,
        Vself,
        beta_self,
        positions=positions,
        dists=dists,
        c=c,
        stochastic=stochastic,
    )


def compute_imitation_return(
    key, choices, weights, Vself, beta_self, positions=None, dists=None, c=0, stochastic=False
):
    # if stochastic, sample agents to imitate from a boltzmann distribution over
    # the weights. otherwise, imitate weight-maximizing agent at every trial
    if stochastic:
        p = boltzmann1d(weights, beta_self)
        targets = random.choice(key, len(weights), shape=(choices.shape[1],), p=p)
    else:
        targets = jnp.argmax(weights)

    # compute imitation return
    imit_choices = choices[targets, jnp.arange(choices.shape[1])]
    return imitation_return(imit_choices, Vself, positions=positions, dists=dists, c=c)


def state_aggregation_experiment(args):
    # initialise results array and prng keys
    results, keys = [], random.split(args.rng_key, args.num_repeats)

    # compute covariance matrix
    K = gp_covariance_matrix(args.side_length, 1, 1)

    # compute entropies for aggregated representations
    agg_factors = jnp.arange(5)
    agg_entropies = jnp.zeros(len(agg_factors))
    for i, af in enumerate(agg_factors):
        patch_length = 2**af
        agg_K = aggregated_covariance_matrix(K, args.side_length, patch_length)
        agg_entropies = agg_entropies.at[i].set(gaussian_entropy(agg_K))
    print(f"aggregated entropies: {agg_entropies}")

    K = gp_covariance_matrix(args.side_length, 1, 2)
    for rep in tqdm(range(args.num_repeats)):
        Vs = sample_value_functions_ind(keys[rep], K, args.side_length, args.num_agents + 1)
        Vself, Vs = Vs[0], Vs[1:]
        choices, positions, dists = get_bandit_choices(
            keys[rep], Vs, args.side_length, args.beta, args.num_trials, args.step_cost
        )

        tmp = simulate_bandit_choices(
            keys[rep],
            Vself,
            args.beta_self,
            args.num_trials,
            positions,
            dists,
            args.step_cost,
        )
        max_ret = imitation_return(
            tmp, Vself, positions=positions, dists=dists, c=args.step_cost
        )

        for i, af in enumerate(agg_factors):
            for policy, rep_name in zip(
                [indiscriminate_policy, individual_agents_policy],
                ["none", "individual"],
            ):
                ret = policy(
                    key=keys[rep],
                    choices=choices,
                    Vself=Vself,
                    Vs=Vs,
                    beta_self=args.beta_self,
                    af=af,
                    positions=positions,
                    dists=dists,
                    c=args.step_cost,
                    stochastic=args.stochastic,
                )

                cost = 0 if rep_name == "none" else args.num_agents * agg_entropies[i]
                results.append(
                    {
                        "return": float(ret / max_ret),
                        "cost": float(cost),
                        "representation": rep_name,
                        "aggregation": float((2**af) ** 2),  # number of states per patch
                    },
                )

    dir_path = f"results/bandit/new/{args.seed}"
    os.makedirs(dir_path, exist_ok=True)

    # normalise return and cost to [0, 1]
    df = pd.DataFrame(results)
    df["return"] = (df["return"] - df["return"].min()) / (
        df["return"].max() - df["return"].min()
    )
    df["cost"] = (df["cost"] - df["cost"].min()) / (df["cost"].max() - df["cost"].min())

    lambdas = [0, 0.1, 0.2, 0.5]
    palette = sns.color_palette("viridis", n_colors=len(lambdas))

    # plot just return
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=df[df["representation"] == "individual"],
        x="aggregation",
        y="return",
        legend=False,
        ax=ax,
        palette=palette,
    )
    sns.lineplot(
        data=df[df["representation"] == "none"],
        x="aggregation",
        y="return",
        ax=ax,
        color="red",
        linestyle="dashed",
        alpha=0.3,
        legend=False,
    )
    ax.set_xscale("log", base=2)
    ax.set_title(
        "Average imitation return using aggregated value function estimates", fontsize=12
    )
    # add legend
    ax.plot([], [], color="blue", label="selective")
    ax.plot([], [], color="red", linestyle="dashed", label="indiscriminate")
    ax.legend(loc="lower right")
    for fmt in ["svg", "png"]:
        fig.savefig(f"{dir_path}/return.{fmt}")

    # plot just cost
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=df[df["representation"] == "individual"],
        x="aggregation",
        y="cost",
        legend=False,
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ax.set_title(
        "Average imitation return using aggregated value function estimates", fontsize=12
    )
    for fmt in ["svg", "png"]:
        fig.savefig(f"{dir_path}/cost.{fmt}")

    # for each value of lambda, copy the dataframe and compute cost-adjusted return
    # then append to list of dataframes and concatenate
    dfs = []
    for l in lambdas:
        df_copy = df.copy()
        df_copy["lambda"] = [l for _ in range(len(df_copy))]
        df_copy["return"] = ((1 - l) * df_copy["return"]) - (l * df_copy["cost"])
        dfs.append(df_copy)
    new_df = pd.concat(dfs, ignore_index=True)

    # plot cost-adjusted return
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=new_df[new_df["representation"] == "individual"],
        x="aggregation",
        y="return",
        hue="lambda",
        ax=ax,
        palette=palette,
    )
    ax.set_xscale("log", base=2)
    ax.set_title(
        "Cost-adjusted average imitation return using aggregated value function estimates",
        fontsize=12,
    )
    for fmt in ["svg", "png"]:
        fig.savefig(f"{dir_path}/combined.{fmt}")


def groups_experiment(args):
    results, keys = [], random.split(args.rng_key, args.num_repeats)

    K = gp_covariance_matrix(args.side_length, 1, 1)
    ind_entropy = gaussian_entropy(K)
    grp_entropy = uniform_entropy(args.num_groups)

    for rho in [10**i for i in range(-4, 3)]:
        for rep in tqdm(range(args.num_repeats), desc=f"rho = {rho}"):
            K = gp_covariance_matrix(args.side_length, 1, 2)
            group_Vs, z, Vs = sample_value_functions_grp(
                keys[rep],
                K,
                args.side_length,
                args.num_groups,
                args.agents_per_group,
                rho,
            )
            Vself = Vs[0]

            choices, positions, dists = get_bandit_choices(
                keys[rep],
                Vs,
                args.side_length,
                args.beta,
                args.num_trials,
                args.step_cost,
            )

            total_agents = args.num_groups * args.agents_per_group

            for policy, rep_name in zip(
                [indiscriminate_policy, individual_agents_policy, group_agents_policy],
                ["none", "individual", "group"],
            ):
                ret = policy(
                    key=keys[rep],
                    choices=choices,
                    Vself=Vself,
                    group_Vs=group_Vs,
                    z=z,
                    Vs=Vs,
                    beta_self=args.beta_self,
                    af=args.af,
                    positions=positions,
                    dists=dists,
                    c=args.step_cost,
                    stochastic=args.stochastic,
                )

                cost = 0
                if rep_name == "individual":
                    cost = total_agents * ind_entropy
                elif rep_name == "group":
                    cost = (args.num_groups * ind_entropy) + (total_agents * grp_entropy)

                results.append(
                    {
                        "agents": total_agents,
                        "return": float(ret),
                        "cost": float(cost),
                        "representation": rep_name,
                        "rho": rho,
                    },
                )

    # dir_path = f"results/bandit/2d/groups/{seed}_{args.rho}"
    dir_path = f"results/bandit/new/groups/{args.seed}"
    os.makedirs(dir_path, exist_ok=True)

    # save results to file
    df = pd.DataFrame(results)
    df.to_pickle(f"{dir_path}/results.pkl")

    # normalise return and cost columns to have max value of 1
    # then compute cost-adjusted return
    df["return"] = df["return"] / df["return"].max()
    df["cost"] = df["cost"] / df["cost"].max()

    lambdas = [0, 0.1, 0.2, 0.5, 1.0]
    palette = sns.color_palette("viridis", n_colors=2)
    fig, axs = plt.subplots(1, len(lambdas), figsize=(16, 4), sharey=True)
    for i, l in enumerate(lambdas):
        df_copy = df.copy()
        df_copy["return"] = ((1 - l) * df_copy["return"]) - (l * df_copy["cost"])

        sns.lineplot(
            data=df_copy[df_copy["representation"] != "none"],
            x="rho",
            y="return",
            hue="representation",
            ax=axs[i],
            palette=palette,
            legend=False,
        )

        sns.lineplot(
            data=df_copy[df_copy["representation"] == "none"],
            x="rho",
            y="return",
            ax=axs[i],
            color="red",
            alpha=0.35,
            linestyle="dashed",
            legend=False,
        )

        if i == 0:
            axs[i].plot([], [], color=palette[0], label="individual")
            axs[i].plot([], [], color=palette[1], label="group")
            axs[i].plot([], [], color="red", alpha=0.35, linestyle="dashed", label="none")
            axs[i].legend(loc="lower right", title="representation")

        axs[i].set_title(f"$\lambda = {l}$", fontsize=11)
        axs[i].set_xlabel(r"$\rho$", fontsize=11)
        axs[i].set_ylabel("cost-adjusted return", fontsize=11)
        axs[i].set_xscale("log", base=10)

    fig.suptitle(
        "Cost-adjusted average imitation return using group representations",
        fontsize=14,
    )
    fig.tight_layout()
    for fmt in ["svg", "png"]:
        fig.savefig(f"{dir_path}/imitation-return.{fmt}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-repeats", type=int, default=100)
    parser.add_argument("--side-length", type=int, default=16)
    parser.add_argument("--af", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument("--num-groups", type=int, default=10)
    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--agents-per-group", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--beta-self", type=float, default=0.01)
    parser.add_argument("--step-cost", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1e-6)
    parser.add_argument("--stochastic", type=int, default=1)
    parser.add_argument("--run-number", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment", type=str, default="agg")

    args = parser.parse_args()

    if args.seed == 0:
        args.seed = time.time_ns() // 1000
    args.rng_key = random.PRNGKey(args.seed)

    # print args
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    if args.experiment == "agg":
        state_aggregation_experiment(args)
    elif args.experiment == "groups":
        groups_experiment(args)
    else:
        raise ValueError(f"Unrecognised experiment: {args.experiment}")

    # group_Vs, z, Vs = sample_groups(rng_key, args.side_length, 3, 5, 1e-6)

    # fig1, axs1 = plt.subplots(1, 3)
    # for i in range(3):
    #     axs1[i].imshow(group_Vs[i])
    #     axs1[i].axis("off")

    # fig2, axs2 = plt.subplots(3, 5)
    # axs2 = axs2.flatten()
    # for i in range(15):
    #     axs2[i].imshow(Vs[i])
    #     axs2[i].axis("off")

    # plt.show()


if __name__ == "__main__":
    main()
