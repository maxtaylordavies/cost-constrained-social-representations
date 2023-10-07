import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(context="paper", style="darkgrid")
pylab.rcParams.update(
    {
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        # "xtick.labelsize": "x-large",
        # "ytick.labelsize": "x-large",
    }
)


def make_condition_barplots(data, flavour="binary", filename=None, format="svg"):
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8, 8))
    for i, vis in enumerate([False, True]):
        df = data[data["own_group_visible"] == vis]

        tmp = df.loc[df["agents_known"] == True]
        barplot(tmp, flavour=flavour, ax=axs[0, i], show_x_label=False, show_y_label=i == 0)

        tmp = df.loc[(df["agents_known"] == False) & (df["groups_relevant"] == True)]
        barplot(tmp, flavour=flavour, ax=axs[1, i], show_x_label=False, show_y_label=i == 0)

        tmp = df.loc[(df["agents_known"] == False) & (df["groups_relevant"] == False)]
        barplot(tmp, flavour=flavour, ax=axs[2, i], show_x_label=True, show_y_label=i == 0)

    fig.tight_layout()

    filename = filename or f"barplots_{flavour}"
    fig.savefig(f"results/{filename}.{format}")


def barplot(data, ax, flavour="binary", show_x_label=True, show_y_label=True):
    if len(data) == 0:
        return

    # estimator = lambda x: sum(x) / len(x) if flavour == "binary" else "mean"
    estimator = "mean"

    x_label = "Agent" if show_x_label else ""
    y_label = "Imitation" if show_y_label else ""

    plot = sns.barplot(
        data=data,
        x="agent",
        y=f"similarity",
        palette=["#01C58A", "#1100D1"],
        estimator=estimator,
        ax=ax,
    )
    plot.axhline(0.5, ls="--", color="black")
    plot.set(ylim=(0, 1), xlabel=x_label, ylabel=y_label)


def surfaceplot(
    data, keys, labels, ticks=None, ticklabels=None, filename="surf", format="svg"
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        data[keys[0]], data[keys[1]], data[keys[2]], cmap=plt.cm.viridis, linewidth=0.2
    )
    ax.set_xlabel(labels[0], fontsize=14)
    ax.set_ylabel(labels[1], fontsize=14)
    ax.set_zlabel(labels[2], fontsize=14)
    ax.view_init(25, 160)

    if ticks:
        if "x" in ticks:
            ax.set_xticks(ticks["x"])
        if "y" in ticks:
            ax.set_yticks(ticks["y"])
        if "z" in ticks:
            ax.set_zticks(ticks["z"])

    if ticklabels:
        if "x" in ticklabels:
            ax.set_xticklabels(ticklabels["x"])
        if "y" in ticklabels:
            ax.set_yticklabels(ticklabels["y"])
        if "z" in ticklabels:
            ax.set_zticklabels(ticklabels["z"])

    fig.savefig(f"{filename}.{format}")


def plot_model_effectiveness(data, weights, phis, vselfs, model_name, filename, use_2d=False):
    if use_2d:
        plot_model_effectiveness_2d(data, weights, phis, vselfs, model_name, filename)
    else:
        plot_model_effectiveness_1d(data, weights, phis, vselfs, model_name, filename)


def plot_model_effectiveness_1d(data, weights, phis, vselfs, filename):
    fig, axs = plt.subplots(3, len(vselfs), figsize=(12, 8), sharex=False, sharey="row")
    palette = sns.color_palette("viridis", n_colors=len(phis))

    for i, vself in enumerate(vselfs):
        df = data[data["vself"] == i]

        # plot histogram of value functions
        sns.histplot(
            data=df,
            stat="probability",
            x="v",
            hue="phi",
            bins=20,
            alpha=0.5,
            zorder=1,
            kde=True,
            line_kws={
                "linewidth": 2.5,
                "alpha": 1.0,
            },
            legend=False,
            ax=axs[0, i],
            palette=palette,
        )
        axs[0, i].set(
            xlabel="$\mathbf{v}$",
            ylabel="$p(\mathbf{v}|\phi)$",
            title=f"$\mathbf{{v}}^{{(self)}}={vself[0]}$",
        )

        # plot imitation rewards
        sns.stripplot(
            data=df,
            x="phi",
            y="reward",
            hue="phi",
            dodge=True,
            alpha=0.2,
            zorder=1,
            legend=False,
            ax=axs[1, i],
            palette=palette,
        )
        sns.pointplot(
            data=df,
            x="phi",
            y="reward",
            hue="phi",
            join=False,
            dodge=0.8 - 0.8 / 3,
            # markers="d",
            scale=1.25,
            errorbar=None,
            ax=axs[1, i],
            palette=palette,
        )
        ylabel = "Imitation reward" if i == 0 else None
        axs[1, i].set(xlabel=None, xticks=[], xticklabels=[], ylabel=ylabel)
        axs[1, i].legend([], [], frameon=False)

        tmp = [
            {"model": k, "phi": phis[j], "w": float(v[i][j])}
            for k, v in weights.items()
            for j in range(len(phis))
        ]
        weights_df = pd.DataFrame(tmp)

        sns.lineplot(
            data=weights_df,
            x="phi",
            y="w",
            style="model",
            markers=True,
            ax=axs[2, i],
            palette=palette,
            legend=i == len(vselfs) - 1,
        )

        # plot computed weights
        # sns.barplot(
        #     x=phis,
        #     y=weights[i],
        #     ax=axs[2, i],
        #     palette=palette,
        # )

        ylabel = "Weight" if i == 0 else None
        axs[2, i].set(ylim=(0, 1), xlabel="$\phi$", ylabel=ylabel)

    fig.suptitle("Comparison of approaches in 1D gridworld", fontsize=16)
    fig.tight_layout()

    plt.show()
    fig.savefig(filename)


def plot_model_effectiveness_2d(data, weights, phis, vselfs, model_name, filename):
    fig, axs = plt.subplots(3, len(vselfs), figsize=(12, 8), sharex=False, sharey="row")
    palette = sns.color_palette("viridis", n_colors=len(phis))

    for i, vself in enumerate(vselfs):
        df = data[data["vself"] == i]

        # plot histogram of value functions
        sns.histplot(
            data=df,
            stat="probability",
            x="vx",
            y="vy",
            hue="phi",
            alpha=0.4,
            bins=30,
            zorder=1,
            legend=False,
            ax=axs[0, i],
            palette=palette,
        )
        sns.kdeplot(
            data=df,
            stat="probability",
            x="vx",
            y="vy",
            hue="phi",
            alpha=0.7,
            zorder=2,
            levels=5,
            legend=False,
            ax=axs[0, i],
            palette=palette,
        )
        axs[0, i].set(
            xlabel="$\mathbf{v}_x$",
            ylabel="$\mathbf{v}_y$",
            title=f"$\mathbf{{v}}^{{(self)}}=[{vself[0]},{vself[1]}]$",
        )

        # add red x at vself
        axs[0, i].scatter(
            vself[0],
            vself[1],
            marker="x",
            color="red",
            s=100,
            linewidth=2.5,
            zorder=3,
        )

        # plot imitation rewards
        sns.stripplot(
            data=df,
            x="phi",
            y="reward",
            hue="phi",
            dodge=True,
            alpha=0.2,
            zorder=1,
            legend=False,
            ax=axs[1, i],
            palette=palette,
        )
        sns.pointplot(
            data=df,
            x="phi",
            y="reward",
            hue="phi",
            join=False,
            dodge=0.8 - 0.8 / 3,
            scale=1.25,
            errorbar=None,
            ax=axs[1, i],
            palette=palette,
        )
        ylabel = "Imitation reward" if i == 0 else None
        axs[1, i].set(xlabel=None, xticks=[], xticklabels=[], ylabel=ylabel)
        axs[1, i].legend([], [], frameon=False)

        # plot computed weights
        sns.barplot(
            x=phis,
            y=weights[i],
            ax=axs[2, i],
            palette=palette,
        )
        ylabel = "Weight" if i == 0 else None
        axs[2, i].set(ylim=(0, 1), xlabel="$\phi$", ylabel=ylabel)

    fig.suptitle(model_name, fontsize=16)
    fig.tight_layout()

    plt.show()
    # fig.savefig(filename)
