
import numpy as np

def hungarian_argmax(sim: np.ndarray):
    """
    Return row/col indices for max-similarity assignment.
    Uses SciPy if available; otherwise greedy fallback.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-sim)  # maximize sim
        return r, c
    except Exception:
        # greedy fallback
        N = sim.shape[0]
        sim_copy = sim.copy()
        r_idx, c_idx = [], []
        for _ in range(N):
            i, j = np.unravel_index(np.argmax(sim_copy), sim_copy.shape)
            if sim_copy[i, j] == -np.inf:
                break
            r_idx.append(i); c_idx.append(j)
            sim_copy[i, :] = -np.inf
            sim_copy[:, j] = -np.inf
        return np.array(r_idx, dtype=int), np.array(c_idx, dtype=int)
