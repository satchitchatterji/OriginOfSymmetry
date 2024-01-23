import matplotlib.pyplot as plt

from util import open_experiment_table

def plot_float_property_over_generations(df, property_name="fitness"):
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
    plt.show()

def main() -> None:
    """Run the program."""
    df = open_experiment_table()
    print(df)
    plot_float_property_over_generations(df, "fitness")
    plot_float_property_over_generations(df, "symmetry")



if __name__ == "__main__":
    main()
