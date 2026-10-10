"""Grouping fired cells into clusters.

A wrong grouping does not raise: it merges or splits showers silently and every
plot downstream still looks reasonable. These tests pin the connectivity rule,
the independence from hit order, and the agreement with the definition
``ShowerAnalyzer`` uses for the ``NClusters`` observable.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdhcal import clustering


def n_clusters(labels: np.ndarray) -> int:
    return 0 if labels.size == 0 else int(labels.max()) + 1


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------
def test_adjacent_cells_in_one_layer_form_one_cluster():
    labels = clustering.cluster_labels(i=[1, 2, 3], j=[5, 5, 5], k=[0, 0, 0])
    assert n_clusters(labels) == 1


def test_a_gap_splits_the_cluster():
    labels = clustering.cluster_labels(i=[1, 2, 4, 5], j=[0, 0, 0, 0], k=[0, 0, 0, 0])
    assert n_clusters(labels) == 2
    assert labels[0] == labels[1] and labels[2] == labels[3]


def test_diagonal_neighbours_are_not_connected():
    """Four-connectivity, as in ShowerAnalyzer::getNeighbors."""
    labels = clustering.cluster_labels(i=[0, 1], j=[0, 1], k=[0, 0])
    assert n_clusters(labels) == 2


def test_cells_in_different_layers_are_never_connected():
    """Clustering is per layer; the longitudinal direction is not a neighbour."""
    labels = clustering.cluster_labels(i=[0, 0], j=[0, 0], k=[0, 1])
    assert n_clusters(labels) == 2


def test_an_l_shape_is_one_cluster():
    labels = clustering.cluster_labels(i=[0, 1, 1, 1], j=[0, 0, 1, 2], k=[0, 0, 0, 0])
    assert n_clusters(labels) == 1


def test_labels_do_not_depend_on_hit_order():
    i, j, k = [0, 1, 5, 6], [0, 0, 0, 0], [0, 0, 0, 0]
    straight = clustering.cluster_labels(i, j, k)
    order = [3, 1, 0, 2]
    shuffled = clustering.cluster_labels([i[q] for q in order], [j[q] for q in order], [k[q] for q in order])
    # Same partition, whatever the order the hits arrive in.
    assert n_clusters(straight) == n_clusters(shuffled) == 2
    for a in range(4):
        for b in range(4):
            assert (straight[a] == straight[b]) == (shuffled[order.index(a)] == shuffled[order.index(b)])


def test_labels_are_consecutive_from_zero():
    labels = clustering.cluster_labels(i=[0, 5, 10], j=[0, 0, 0], k=[0, 0, 0])
    assert sorted(set(labels.tolist())) == [0, 1, 2]


def test_empty_and_mismatched_inputs():
    assert clustering.cluster_labels([], [], []).size == 0
    with pytest.raises(ValueError, match="same length"):
        clustering.cluster_labels(i=[0, 1], j=[0], k=[0])


def test_a_large_random_shower_is_grouped_consistently():
    """Union-find must agree with a plain flood fill on a realistic occupancy."""
    rng = np.random.default_rng(0)
    k = rng.integers(0, 48, 2000)
    i = rng.integers(0, 40, 2000)
    j = rng.integers(0, 40, 2000)
    cells = sorted(set(zip(k.tolist(), i.tolist(), j.tolist(), strict=True)))
    k, i, j = (np.array([c[q] for c in cells]) for q in range(3))

    labels = clustering.cluster_labels(i, j, k)

    occupied = {(int(k[q]), int(i[q]), int(j[q])): q for q in range(len(i))}
    seen, expected = set(), 0
    for start in range(len(i)):
        if start in seen:
            continue
        expected += 1
        stack = [start]
        seen.add(start)
        while stack:
            q = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                other = occupied.get((int(k[q]), int(i[q]) + di, int(j[q]) + dj))
                if other is not None and other not in seen:
                    seen.add(other)
                    stack.append(other)
    assert n_clusters(labels) == expected


# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------
def test_centroid_size_and_threshold_fractions():
    labels = np.array([0, 0, 1])
    features = clustering.cluster_features(
        x=[0.0, 2.0, 10.0],
        y=[0.0, 4.0, 10.0],
        z=[1.0, 1.0, 5.0],
        time=[7.0, 3.0, 9.0],
        threshold=[1, 3, 2],
        labels=labels,
    )
    names = clustering.CLUSTER_FEATURE_NAMES
    first = dict(zip(names, features[0], strict=True))
    assert first["x"] == pytest.approx(1.0)  # centroid of the two pads
    assert first["y"] == pytest.approx(2.0)
    assert first["time"] == pytest.approx(3.0)  # earliest, not mean
    assert first["size"] == pytest.approx(2.0)
    assert first["frac_thr1"] == pytest.approx(0.5)
    assert first["frac_thr3"] == pytest.approx(0.5)
    assert first["frac_thr2"] == pytest.approx(0.0)

    second = dict(zip(names, features[1], strict=True))
    assert second["size"] == pytest.approx(1.0)
    assert second["frac_thr2"] == pytest.approx(1.0)


def test_threshold_fractions_sum_to_one():
    rng = np.random.default_rng(3)
    n = 200
    labels = rng.integers(0, 20, n)
    labels = np.unique(labels, return_inverse=True)[1]
    features = clustering.cluster_features(
        x=rng.normal(size=n),
        y=rng.normal(size=n),
        z=rng.normal(size=n),
        time=rng.normal(size=n),
        threshold=rng.integers(1, 4, n),
        labels=labels,
    )
    index = {name: q for q, name in enumerate(clustering.CLUSTER_FEATURE_NAMES)}
    total = sum(features[:, index[f"frac_thr{level}"]] for level in (1, 2, 3))
    assert total == pytest.approx(np.ones(len(features)))


def test_sizes_sum_back_to_the_hit_count():
    labels = np.array([0, 0, 0, 1, 2, 2])
    features = clustering.cluster_features(
        x=np.zeros(6),
        y=np.zeros(6),
        z=np.zeros(6),
        time=np.zeros(6),
        threshold=np.ones(6, dtype=int),
        labels=labels,
    )
    size = features[:, clustering.CLUSTER_FEATURE_NAMES.index("size")]
    assert size.sum() == pytest.approx(6.0)


def test_empty_input_gives_an_empty_table():
    features = clustering.cluster_features([], [], [], [], [], np.empty(0, dtype=np.int64))
    assert features.shape == (0, len(clustering.CLUSTER_FEATURE_NAMES))
