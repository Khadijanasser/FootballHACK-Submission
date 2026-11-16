"""Deterministic Viterbi decoding of H/A/D probabilities within one half."""
import numpy as np

STATES = np.array(["H", "A", "D"])


def viterbi(proba, *, turnover=0.0, stop=0.0, restart=0.0, floor=1e-4):
    """Most likely state path with additive log-domain transition penalties.

    turnover: H<->A cost; stop: H/A->D cost; restart: D->H/A cost. All zero
    penalties reduce to the per-frame argmax. No minimum duration is imposed.
    """
    p = np.asarray(proba, dtype=float)
    n = len(p)
    if n == 0:
        return np.empty(0, dtype="U1")
    emit = np.log(np.clip(p, floor, 1.0))
    cost = np.array([[0.0, turnover, stop], [turnover, 0.0, stop], [restart, restart, 0.0]])
    score = emit[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        candidate = score[:, None] - cost          # from-state rows, to-state columns
        best = np.argmax(candidate, axis=0)        # ties resolve to the lowest state index
        back[t] = best
        score = candidate[best, (0, 1, 2)] + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return STATES[path]
