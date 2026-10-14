"""
Transition ("circle") graphs for BL vs FST.

Faithful reimplementation of keypoint-MoSeq's transition-graph pipeline, so these
match the CalEx-vs-Control graphs already in the paper. Every step below is ported
from keypoint_moseq/analysis.py:

  get_transitions        locs = where(arr[1:] != arr[:-1]) + 1;  transitions = arr[locs]
                         -> the BOUT sequence, not the frame sequence
  n_gram_transition_matrix   bigram counts over that bout sequence, max_label=100
  get_transition_matrix(combine=True)   sum counts across recordings, then normalise once
  normalize="bigram"     matrix /= matrix.sum()
  syll_include           pooled bout-frequency > min_frequency, across ALL recordings
  usages                 per-group bout frequency, used for node area
  plot_transition_graph_group   nx.from_numpy_array(mat*100), circular layout,
                         node_size = usage/sum*2000 + 1000, white nodes / red edges,
                         black edges at alpha 0.6, figsize (20, 10*rows), 2 columns

Outputs (results/transition_graphs/), each at two line-width scalings, {thin,thick}:
  transition_ALL_*.pdf                 all included syllables, BL and FST side by side
  transition_DIFFERENCE_*.pdf          FST minus BL; red = higher in FST, blue = higher in BL
  transition_allogrooming_*.pdf        edges incident on syllables 5 + 19 only
  transition_allogrooming_DIFF_*.pdf   the same, as a difference graph
  transition_selfgrooming_*.pdf        edges incident on syllables 9, 10, 11 only
  transition_selfgrooming_DIFF_*.pdf   the same, as a difference graph
  transition_strengths.csv             per-group bout share and transition weight for
                                       every syllable of interest

Usage:
    python scripts/transition_graphs.py
"""
import os
from math import ceil

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_H5 = os.path.join(ROOT, "data", "model", "results.h5")
GROUPS_CSV = os.path.join(ROOT, "data", "groups.csv")
OUT = os.path.join(ROOT, "results", "transition_graphs")

MAX_LABEL = 100
MIN_FREQUENCY = 0.005
SELECTED = ["BL_subject", "FST_subject"]

FOCUS_SETS = [
    ("allogrooming", [5, 19]),
    ("selfgrooming", [9, 10, 11]),
]


# ── ported from keypoint_moseq / jax_moseq ───────────────────────────────
def get_transitions(seq):
    seq = np.asarray(seq)
    locs = np.where(seq[1:] != seq[:-1])[0] + 1
    return seq[locs]


def n_gram_transition_matrix(transitions, max_label=MAX_LABEL):
    m = np.zeros((max_label, max_label), dtype=float)
    for a, b in zip(transitions[:-1], transitions[1:]):
        if a >= max_label or b >= max_label:
            continue
        m[a, b] += 1
    return m


def get_frequencies(seqs, num_states=MAX_LABEL):
    """runlength=True — share of bouts, the keypoint-MoSeq default."""
    flat = np.concatenate([np.asarray(s) for s in seqs]).astype(int)
    onsets = np.pad(np.diff(flat).nonzero()[0] + 1, (1, 0))
    counts = np.bincount(flat[onsets], minlength=num_states)
    return counts / counts.sum()


def group_trans_mats(labels, label_group, groups, syll_include):
    mats, usages = [], []
    for g in groups:
        use = [l for l, grp in zip(labels, label_group) if grp == g]
        counts = np.sum([n_gram_transition_matrix(get_transitions(v)) for v in use],
                        axis=0)
        norm = counts / counts.sum()                      # normalize="bigram"
        mats.append(norm[syll_include, :][:, syll_include])
        usages.append(get_frequencies(use)[syll_include])
    return mats, usages


def build():
    meta = pd.read_csv(GROUPS_CSV)
    meta["group"] = meta["condition"] + "_" + meta["role"]
    with h5py.File(RESULTS_H5, "r") as f:
        labels = [f[n]["syllable"][:] for n in meta["name"]]

    groups = sorted(meta["group"].unique())
    print("Group(s):", ", ".join(groups))

    # syll_include is computed across ALL recordings, exactly as kpms does
    syll_include = np.where(get_frequencies(labels) > MIN_FREQUENCY)[0]
    mats, usages = group_trans_mats(labels, list(meta["group"]), groups, syll_include)

    idx = {g: i for i, g in enumerate(groups)}
    sel = [idx[g] for g in SELECTED]
    return ([groups[i] for i in sel], [mats[i] for i in sel],
            [usages[i] for i in sel], syll_include)


# ── plotting ─────────────────────────────────────────────────────────────
def draw(groups, mats, usages, syll_include, path, focus=None, title_suffix="",
         width_mode="raw", max_line_width=10.0, min_line_width=0.0, edge_alpha=0.5):
    """Circular transition graph, 20x10 for a two-panel row (kpms dimensions).

    width_mode="raw"    line width = the edge weight itself, i.e. what
                        plot_transition_graph_group does -> the THIN version
    width_mode="scaled" line width = |w| / global_max * max_line_width, i.e. what
                        plot_transition_graph_group_fixedstyle does -> the THICK version

    focus=None draws every edge. A list of syllables keeps only the edges touching
    them; thicknesses are normalised against the FULL graph either way, so a focused
    figure can be compared against the complete one without rescaling.
    """
    names = [str(s) for s in syll_include]
    n_row = ceil(len(groups) / 2)
    fig, all_axes = plt.subplots(n_row, 2, figsize=(20, 10 * n_row))
    ax = np.atleast_1d(all_axes).flat

    # global max always comes from the FULL matrices, never the focused subset
    w_all = [np.abs(list(nx.get_edge_attributes(
        nx.from_numpy_array(m * 100), "weight").values())) for m in mats]
    global_max = max((x.max() for x in w_all if len(x)), default=1.0) or 1.0

    focus_idx = ([i for i, s in enumerate(syll_include) if s in focus]
                 if focus is not None else [])

    for i, g in enumerate(groups):
        mat = mats[i]
        if focus is not None:
            keep = np.zeros_like(mat)
            for k in focus_idx:
                keep[k, :] = mat[k, :]
                keep[:, k] = mat[:, k]
            mat = keep

        G = nx.from_numpy_array(mat * 100)
        widths = nx.get_edge_attributes(G, "weight")
        pos = nx.circular_layout(G)
        nodelist = G.nodes()

        node_size = np.array(usages[i]) / sum(usages[i]) * 2000 + 1000

        if width_mode == "raw":
            wvals = list(widths.values())
        else:
            v = np.abs(list(widths.values()))
            wvals = np.clip(v / global_max * max_line_width, min_line_width, None)

        node_color = ["lightcoral" if j in focus_idx else "white" for j in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=node_size,
                               node_color=node_color, edgecolors="red", ax=ax[i])
        if widths:
            nx.draw_networkx_edges(G, pos, edgelist=widths.keys(), width=wvals,
                                   edge_color="black", ax=ax[i],
                                   alpha=0.6 if width_mode == "raw" else edge_alpha)
        nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(nodelist, names)),
                                font_color="black", ax=ax[i])
        ax[i].set_title(f"{g}{title_suffix}", fontsize=14)
        ax[i].set_xlim(-1.2, 1.2)
        ax[i].set_ylim(-1.2, 1.2)
        ax[i].set_aspect("equal")

    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150,
                facecolor="white")
    plt.close(fig)
    print(f"  {os.path.basename(path)}")


def draw_difference(groups, mats, usages, syll_include, path, node_scaling=3000,
                    width_mode="raw", max_line_width=10.0, focus=None,
                    title_suffix=""):
    """Port of plot_transition_graph_difference.

    kpms computes tm_diff = trans_mats[left] - trans_mats[right] and colours an edge
    "blue" if the difference is POSITIVE, "red" otherwise, while labelling red
    "Up-regulated" in the legend. Net effect: red marks transitions that are HIGHER IN
    THE SECOND GROUP of the title. Groups are ordered BL then FST below, so red = up in
    FST. The legend here spells that out rather than repeating the ambiguous wording.
    """
    from matplotlib.lines import Line2D

    left, right = 0, 1                                  # BL_subject, FST_subject
    tm_diff_full = mats[left] - mats[right]
    usages_diff = np.array(usages[left]) - np.array(usages[right])
    node_size = (np.abs(usages_diff) / np.abs(usages_diff).sum()) * node_scaling + 500

    # Scale against the FULL difference matrix, so a focused figure keeps the same
    # thickness scale as the complete one and the two can be read side by side.
    full_max = np.abs(tm_diff_full * 1000).max() or 1.0

    focus_idx = ([i for i, s in enumerate(syll_include) if s in focus]
                 if focus is not None else [])
    tm_diff = tm_diff_full
    if focus is not None:
        keep = np.zeros_like(tm_diff_full)
        for k in focus_idx:
            keep[k, :] = tm_diff_full[k, :]
            keep[:, k] = tm_diff_full[:, k]
        tm_diff = keep

    G = nx.from_numpy_array(tm_diff * 1000)
    pos = nx.circular_layout(G)
    widths = nx.get_edge_attributes(G, "weight")
    names = [str(s) for s in syll_include]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    node_face = ["#ffe9e9" if j in focus_idx else "white" for j in G.nodes()]
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=node_size,
                           node_color=node_face,
                           edgecolors=["blue" if u > 0 else "red" for u in usages_diff],
                           linewidths=[2.5 if j in focus_idx else 1.0 for j in G.nodes()],
                           ax=ax)
    v = np.abs(list(widths.values()))
    wvals = v if width_mode == "raw" else v / full_max * max_line_width
    nx.draw_networkx_edges(G, pos, edgelist=widths.keys(), width=wvals,
                           edge_color=["blue" if u > 0 else "red"
                                       for u in widths.values()],
                           ax=ax, alpha=0.6)
    nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(G.nodes(), names)),
                            font_color="black", ax=ax)
    ax.set_title(f"{groups[left]} - {groups[right]}{title_suffix}")
    ax.set_aspect("equal")          # kpms leaves this free; the legend used to fill
    ax.axis("off")                  # the right half and fake a circle. Force it.
    ax.legend(handles=[
        Line2D([0], [0], color="r", lw=2, label="Transition higher in FST"),
        Line2D([0], [0], color="b", lw=2, label="Transition higher in BL"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
               markeredgecolor="r", markersize=10, label="Usage higher in FST"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
               markeredgecolor="b", markersize=10, label="Usage higher in BL"),
    ], loc="upper right", bbox_to_anchor=(1.32, 1.0))

    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150,
                facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")

    for s in (focus if focus is not None else [5]):
        k = list(syll_include).index(s)
        net = tm_diff_full[k, :].sum() + tm_diff_full[:, k].sum()
        print(f"    syllable {s}: net diff (BL-FST) = {net:+.4f} -> "
              f"{'red, higher in FST' if net < 0 else 'blue, higher in BL'}")


def main():
    os.makedirs(OUT, exist_ok=True)
    groups, mats, usages, syll_include = build()
    print(f"syllables included (bout freq > {MIN_FREQUENCY}): {syll_include.tolist()}\n")

    # Two line-width variants of everything. Both use the fixedstyle normalisation
    # (|w| / global_max * max_line_width) so the only difference is the cap — that
    # keeps the two versions honestly comparable. Using kpms's raw weights instead
    # would give a max of ~14.6 here, i.e. THICKER than a max_line_width=10 cap, so
    # "raw vs capped" is not reliably a thin/thick pair.
    for tag, mlw in [("thin", 6.0), ("thick", 15.0)]:
        print(f"\n[{tag} lines — max_line_width={mlw:g}]")
        draw(groups, mats, usages, syll_include,
             os.path.join(OUT, f"transition_ALL_{tag}.pdf"),
             width_mode="scaled", max_line_width=mlw)

        draw_difference(groups, mats, usages, syll_include,
                        os.path.join(OUT, f"transition_DIFFERENCE_{tag}.pdf"),
                        width_mode="scaled", max_line_width=mlw)

        for name, sylls in FOCUS_SETS:
            present = [s for s in sylls if s in syll_include]
            missing = [s for s in sylls if s not in syll_include]
            if missing:
                print(f"  NOTE: {name} syllables {missing} are below the "
                      f"{MIN_FREQUENCY:.1%} threshold and absent from the graph")
            # side-by-side BL / FST panels
            draw(groups, mats, usages, syll_include,
                 os.path.join(OUT, f"transition_{name}_{tag}.pdf"),
                 focus=present, title_suffix=f" — {name} {present}",
                 width_mode="scaled", max_line_width=mlw)
            # red/blue difference version of the same focus set
            draw_difference(groups, mats, usages, syll_include,
                            os.path.join(OUT, f"transition_{name}_DIFF_{tag}.pdf"),
                            width_mode="scaled", max_line_width=mlw,
                            focus=present, title_suffix=f" — {name} {present}")

    # transition strengths involving the focus syllables, as a table
    rows = []
    for name, sylls in FOCUS_SETS:
        for s in [x for x in sylls if x in syll_include]:
            k = list(syll_include).index(s)
            for i, g in enumerate(groups):
                rows.append({"set": name, "syllable": s, "group": g,
                             "outgoing": mats[i][k, :].sum(),
                             "incoming": mats[i][:, k].sum(),
                             "self": mats[i][k, k]})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "transition_strengths.csv"), index=False)
    print(f"\n=== transition probability mass by syllable ===\n")
    print(t.pivot_table(index=["set", "syllable"], columns="group",
                        values=["outgoing", "incoming"]).round(4).to_string())
    print(f"\nAll files in {OUT}")


if __name__ == "__main__":
    main()
