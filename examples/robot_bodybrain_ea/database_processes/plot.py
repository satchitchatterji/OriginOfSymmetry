import matplotlib.pyplot as plt

from util import open_experiment_table
from balance_compute import compute_balance_from_str_list

def plot_float_property_over_generations(df, property_name="fitness", savename=None):
    """Plot property_name (fitness/symmetry/balance) over generations for all experiments, averaged."""
    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({property_name: ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        f"max_{property_name}",
        f"mean_{property_name}",
    ]

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({f"max_{property_name}": ["mean", "std"], f"mean_{property_name}": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index",
        f"max_{property_name}_mean",
        f"max_{property_name}_std",
        f"mean_{property_name}_mean",
        f"mean_{property_name}_std",
    ]

    plt.figure()

    # Plot max
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation[f"max_{property_name}_mean"],
        label=f"Max {property_name}",
        color="b",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation[f"max_{property_name}_mean"] - agg_per_generation[f"max_{property_name}_std"],
        agg_per_generation[f"max_{property_name}_mean"] + agg_per_generation[f"max_{property_name}_std"],
        color="b",
        alpha=0.2,
    )

    # Plot mean
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation[f"mean_{property_name}_mean"],
        label=f"Mean {property_name}",
        color="r",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation[f"mean_{property_name}_mean"]
        - agg_per_generation[f"mean_{property_name}_std"],
        agg_per_generation[f"mean_{property_name}_mean"]
        + agg_per_generation[f"mean_{property_name}_std"],
        color="r",
        alpha=0.2,
    )

    plt.xlabel("Generation index")
    plt.ylabel(f"{property_name}")
    plt.title(f"Mean and max {property_name} across repetitions with std as shade")
    plt.legend()
    plt.grid()
    if savename is not None:
        plt.savefig(savename, bbox_inches="tight")
    plt.show()

def plot_all_properties_single_experiment(df, experiment_id):
    """Plot fitness/symmetry/balance over generations for a single experiment."""
    df = df[df["experiment_id"] == experiment_id]
    plot_float_property_over_generations(df, "fitness", f"fitness_exp{experiment_id}.png")
    plot_float_property_over_generations(df, "symmetry", f"symmetry_exp{experiment_id}.png")
    plot_float_property_over_generations(df, "balance", f"balance_exp{experiment_id}.png")

def preprocess_df(df):
    """Preprocess the dataframe."""
    df = df[df["experiment_id"] >= 11]
    print("*"*80)
    print("Warning: only using experiments with id >= 11. Change this if you want to use other experiments.")
    print("*"*80)
    df["balance"] = compute_balance_from_str_list(df["xy_positions"])
    return df

def main() -> None:
    """Run the program."""
    df = preprocess_df(open_experiment_table())
    print(df)
    for experiment_id in df["experiment_id"].unique():
        plot_all_properties_single_experiment(df, experiment_id)

if __name__ == "__main__":
    main()
