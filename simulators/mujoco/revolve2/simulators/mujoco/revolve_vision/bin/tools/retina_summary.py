#!/usr/bin/env python3
import json
import pprint
import pickle
from pathlib import Path
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def main():
    data = []
    hist_data, time_data = {}, {}
    groups = set()
    budgets, iterations = set(), set()
    base_folder = Path("remote/collect_v3")
    for folder in base_folder.glob("*100K"):
        file = folder.joinpath("best.json")
        if not file.exists():
            continue

        with open(file, 'r') as fd:
            d = json.load(fd)

        rid, group, _ = folder.stem.split('-')
        groups.add(group)
        d.update({"group": group})
        data.append(d)

        def _append(lhs, rhs):
            for k_, v_ in rhs.items():
                if k_ not in lhs:
                    lhs[k_] = []
                lhs[k_].append(v_)

        _append(hist_data, dict(
            rid=rid, group=group,
            gid=d["id"],
            fitness=d["fitnesses"]["collect"],
            vision=reduce(lambda x, y: x * y, d["genome"]["vision"].values()),
            depth=d["stats"]["depth"],
            n_hidden=d["stats"]["hidden"],
            n_total=d["stats"]["neurons"],
            n_links=d["stats"]["edges"],
            cppn_lid=d["genome"]["brain"]["LID"],
            cppn_nid=d["genome"]["brain"]["NID"],
            cppn_ln=len(d["genome"]["brain"]["links"]),
            cppn_nn=len(d["genome"]["brain"]["nodes"]),
        ))

        file = folder.joinpath("iteration-final.p")
        with open(file, 'rb') as fd:
            d = pickle.load(fd)

        budgets.add(d["budget"])
        iterations.add(len(d["iterations"]))

        def _append(lhs, rhs):
            for k_, v_ in rhs.items():
                if k_ not in lhs:
                    lhs[k_] = {}
                if group not in lhs[k_]:
                    lhs[k_][group] = v_
                else:
                    lhs[k_][group] = np.c_[lhs[k_][group], v_]

        def _fmt(k, f): return [f(v_) for v_ in d["iterations"][k]]
        def _norm(l_): return 100 * float(l_[0]) / float(l_[1])
        _append(time_data, dict(
            cont_size=_fmt("cont_size", lambda o: _norm(str(o).split("/"))),
            max=_fmt("max", lambda o: float(json.loads(o)[0])),
            qd_score=_fmt("qd_score", float),
        ))

    groups = sorted(groups)

    assert len(budgets) == 1
    budget = next(iter(budgets))
    assert len(iterations) == 1
    iterations = next(iter(iterations))
    index = np.linspace(budget/iterations, budget, iterations)

    # pprint.pprint(data)
    df = pd.DataFrame(data=hist_data)
    df.to_csv(base_folder.joinpath("hist.csv"), sep=' ', index=False)
    # pprint.pprint(df)

    for c in df.columns[2:]:
        v_min, v_max = np.quantile(df[c], [0, 1])
        hist = df.hist(column=c, by='group', sharex=True, sharey=True,
                       layout=(len(groups), 1),
                       bins=10, range=(v_min, v_max))

        gdf = df.groupby("group")[c]
        for i, g_lhs in enumerate(groups):
            for g_rhs in groups[i+1:]:
                _, p = mannwhitneyu(gdf.get_group(g_lhs), gdf.get_group(g_rhs))
                if p <= .05:
                    print(f"mannwhitneyu({c:10s}, {g_lhs}, {g_rhs}): p={p}")
        hist[-1].set_xlabel(c)
        plt.savefig(base_folder.joinpath(f"hist_{c}.png"))

    # pprint.pprint(time_data)
    variables = list(time_data.keys())
    print(f"{variables=}")
    print(f"{groups=}")
    for v in variables:
        fig, axes = plt.subplots(len(groups), 1, sharex='all', sharey='all')
        fig.supxlabel("Evaluations")
        fig.supylabel(v)
        for i, g in enumerate(groups):
            ax = axes[i]
            ax.set_title(g)
            ax.set_axisbelow(True)
            ax.yaxis.grid(linestyle='dashed')
            d = time_data[v][g]
            # pprint.pprint(d)
            qtls = np.quantile(d, [0, .05, .25, .5, .75, .95, 1], axis=1)
            for j in range(3):
                ax.fill_between(index, qtls[j], qtls[-j-1],
                                alpha=.25, color="C0")
            ax.plot(index, qtls[3], color="C0")

        fig.tight_layout()
        fig.savefig(base_folder.joinpath(f"time_{v}.png"))


if __name__ == '__main__':
    main()
