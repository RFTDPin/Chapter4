
import numpy as np
from typing import Tuple


def topk_accuracy(sim: np.ndarray, k: int = 1) -> float:
    """
    sim: [N, N] similarity matrix, row i is query (video), col j is candidate (AIS).
    Top-k accuracy: fraction of rows whose GT index (i) is within top-k of row i.
    """
    N = sim.shape[0]
    # argsort descending per row
    idx = np.argsort(-sim, axis=1)[:, :k]
    hits = np.any(idx == np.arange(N)[:, None], axis=1).astype(np.float32)
    return float(hits.mean())


def roc_auc(sim: np.ndarray) -> float:
    """
    Compute ROC-AUC from similarity matrix by treating (i,i) as positives and others as negatives.
    Pure NumPy implementation via ranking (equivalent to Mann–Whitney U / AUC).
    """
    N = sim.shape[0]
    # pos = sim[np.arange(N), np.arange(N)].reshape(-1, 1)  # [N,1]
    pos = sim[np.arange(N), np.arange(N)].reshape(-1)  # [N]
    neg = sim.copy()
    np.fill_diagonal(neg, -np.inf)
    # neg = neg.reshape(-1, 1)
    neg = neg.reshape(-1)
    neg = neg[np.isfinite(neg)]  # remove self-pairs

    # scores = np.concatenate([pos, neg], axis=0).reshape(-1)
    # labels = np.concatenate([np.ones_like(pos.reshape(-1)), np.zeros_like(neg.reshape(-1))])
    scores = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    # Compute ranks (average ties)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties
    unique_scores, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_ranks = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_ranks[inv]

    sum_pos_ranks = ranks[labels == 1].sum()
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg + 1e-12)
    return float(auc)


def greedy_match_accuracy(sim: np.ndarray) -> float:
    """
    Greedy 1-1 assignment by repeatedly picking the largest remaining similarity.
    This is a fallback when Hungarian is unavailable. Returns ID assignment accuracy.
    """
    N = sim.shape[0]
    sim_copy = sim.copy()
    rows_assigned = np.full(N, False)
    cols_assigned = np.full(N, False)
    assignment = -np.ones(N, dtype=int)

    for _ in range(N):
        i, j = np.unravel_index(np.argmax(sim_copy), sim_copy.shape)
        if sim_copy[i, j] == -np.inf:
            break
        assignment[i] = j
        rows_assigned[i] = True
        cols_assigned[j] = True
        sim_copy[i, :] = -np.inf
        sim_copy[:, j] = -np.inf

    correct = (assignment == np.arange(N)).astype(np.float32)
    correct[assignment < 0] = 0.0
    return float(correct.mean())


def hungarian_id_accuracy(sim: np.ndarray) -> float:
    """
    If SciPy is available, use linear_sum_assignment on cost = -sim to maximize similarity.
    Otherwise fall back to greedy_match_accuracy.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        cost = -sim
        r, c = linear_sum_assignment(cost)
        # Map row -> assigned col
        assignment = np.full(sim.shape[0], -1, dtype=int)
        assignment[r] = c
        correct = (assignment == np.arange(sim.shape[0])).astype(np.float32)
        return float(correct.mean())
    except Exception:
        return greedy_match_accuracy(sim)


def mofa(sim: np.ndarray) -> float:
    """
    Simplified Multi-Object Fusion Accuracy proxy on similarity matrix:
    ratio of correct 1-1 assignments when maximizing similarity.
    """
    return hungarian_id_accuracy(sim)
