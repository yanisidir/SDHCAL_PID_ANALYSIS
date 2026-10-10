"""Group fired cells into clusters, and summarise each cluster as one node.

A hit-level network pays for every node at every epoch. Grouping neighbouring
fired pads within a layer first, and treating each group as a single node, cuts
the count by about a factor of four on these showers - 753 hits become 191
clusters - without discarding the information, since the cluster keeps its
position, its size and its threshold composition.

That is also what the DHCAL graph study (arXiv:2501.00128) does: "fired
neighboring pads were grouped into clusters, with the cluster's position
determined as the average of the positions of all individual pads". Their
reported factor-of-three saving comes largely from this step rather than from
the choice of architecture.

The grouping is the same one ``ShowerAnalyzer::computeClusters`` uses for the
``NClusters`` observable: four-connectivity in ``(I, J)``, within one layer ``K``.
Keeping the two definitions identical means the shower variable and the network
input describe the same objects.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cluster_labels",
    "cluster_features",
    "cluster_hit_frame",
    "CLUSTER_FEATURE_NAMES",
]

#: What one cluster carries into the network, in this order.
CLUSTER_FEATURE_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "time",
    "size",
    "frac_thr1",
    "frac_thr2",
    "frac_thr3",
)

#: Four-connectivity within a layer, as in ShowerAnalyzer.
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def cluster_labels(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Label each hit with the index of the cluster it belongs to.

    Union-find over the four neighbours of each cell, which visits every hit
    once: the alternative, comparing all pairs, is quadratic and these showers
    reach a thousand hits.

    Labels are consecutive from zero and ordered by first appearance, so the
    result does not depend on how the hits happened to be stored.
    """
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    if not (len(i) == len(j) == len(k)):
        raise ValueError(f"I, J and K must have the same length, got {len(i)}, {len(j)}, {len(k)}")
    n = len(i)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # One cell may legitimately appear once only; a duplicate would mean the same
    # pad fired twice in one event, which the digitisation does not produce.
    position = {}
    for index in range(n):
        position[(int(k[index]), int(i[index]), int(j[index]))] = index

    for index in range(n):
        layer, cell_i, cell_j = int(k[index]), int(i[index]), int(j[index])
        for di, dj in _NEIGHBOURS:
            other = position.get((layer, cell_i + di, cell_j + dj))
            if other is not None:
                union(index, other)

    roots = np.fromiter((find(x) for x in range(n)), dtype=np.int64, count=n)
    _, labels = np.unique(roots, return_inverse=True)
    # np.unique sorts by root index, which is the first hit of each cluster, so
    # the labels already follow first appearance.
    return labels.astype(np.int64)


def cluster_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time: np.ndarray,
    threshold: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """One row per cluster: centroid, earliest time, size, threshold fractions.

    The position is the unweighted centroid of the pads, matching the DHCAL
    definition. The time is the earliest hit of the cluster rather than the mean,
    because a cluster is one crossing and its leading edge is what carries the
    timing information; averaging would blur it with the trailing pads.

    The threshold fractions, rather than counts, keep the composition separate
    from the size, which is already its own feature.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return np.empty((0, len(CLUSTER_FEATURE_NAMES)), dtype=np.float32)

    n_clusters = int(labels.max()) + 1
    threshold = np.asarray(threshold)
    size = np.bincount(labels, minlength=n_clusters).astype(np.float64)

    def mean_of(values: np.ndarray) -> np.ndarray:
        return np.bincount(labels, weights=np.asarray(values, dtype=np.float64), minlength=n_clusters) / size

    earliest = np.full(n_clusters, np.inf)
    np.minimum.at(earliest, labels, np.asarray(time, dtype=np.float64))

    fractions = [
        np.bincount(labels, weights=(threshold == level).astype(np.float64), minlength=n_clusters) / size
        for level in (1, 2, 3)
    ]

    return np.column_stack([mean_of(x), mean_of(y), mean_of(z), earliest, size, *fractions]).astype(
        np.float32
    )


def cluster_hit_frame(frame, *, group_column: str = "event_id"):
    """Group a per-hit table into a per-cluster one, event by event.

    Used by the hit-level PID studies, which read a flat CSV rather than the
    ROOT shards. The output keeps the column names the preprocessor already
    expects - ``x_coords``, ``y_coords``, ``z_coords``, ``time``, ``thr`` - so
    nothing downstream has to know whether a row is a pad or a cluster, and adds
    ``frac_thr1..3`` and ``size``, which is the part a single pad cannot carry.

    ``thr`` on a cluster is the highest threshold it contains. It is kept only so
    that code written for hits still runs; the composition is the real quantity
    and is what the preprocessor uses when it finds it.
    """
    import pandas as pd

    required = {"I", "J", "K", "x_coords", "y_coords", "z_coords", "time", "thr", group_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"cannot cluster: the hit table lacks {', '.join(missing)}. "
            "Re-export it with tools/export_hits.py, which writes the cell indices."
        )
    # Columns that describe the event rather than the hit travel unchanged.
    carried = [c for c in ("label", "primaryEnergy", "shardIndex", "eventNumber") if c in frame.columns]

    rows = []
    for key, hits in frame.groupby(group_column, sort=False):
        labels = cluster_labels(hits["I"].to_numpy(), hits["J"].to_numpy(), hits["K"].to_numpy())
        summary = cluster_features(
            hits["x_coords"].to_numpy(),
            hits["y_coords"].to_numpy(),
            hits["z_coords"].to_numpy(),
            hits["time"].to_numpy(),
            hits["thr"].to_numpy(),
            labels,
        )
        names = CLUSTER_FEATURE_NAMES
        table = pd.DataFrame(
            {
                "x_coords": summary[:, names.index("x")],
                "y_coords": summary[:, names.index("y")],
                "z_coords": summary[:, names.index("z")],
                "time": summary[:, names.index("time")],
                "size": summary[:, names.index("size")],
                "frac_thr1": summary[:, names.index("frac_thr1")],
                "frac_thr2": summary[:, names.index("frac_thr2")],
                "frac_thr3": summary[:, names.index("frac_thr3")],
            }
        )
        highest = np.zeros(len(table), dtype=np.int64)
        np.maximum.at(highest, labels, hits["thr"].to_numpy().astype(np.int64))
        table["thr"] = highest
        table[group_column] = key
        for column in carried:
            table[column] = hits[column].iloc[0]
        rows.append(table)

    return pd.concat(rows, ignore_index=True)
