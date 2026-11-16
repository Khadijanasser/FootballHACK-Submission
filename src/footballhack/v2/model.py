"""Two gradient-boosted classifiers: dead/live, and home/away given live."""
import numpy as np

from .features import build_features, swap_teams

SEED = 2027
HGB = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, early_stopping=False, random_state=SEED)


def half_features(player_xy, ball_xy, teams, pitch_m, *, fps=25.0, normalized=False):
    """Features for the given team labelling and for the swapped labelling."""
    X, names = build_features(player_xy, ball_xy, teams, pitch_m, fps=fps, normalized=normalized)
    Xs, _ = build_features(player_xy, ball_xy, swap_teams(teams), pitch_m, fps=fps, normalized=normalized)
    return X, Xs, names


def fit(X, Xs, labels):
    """labels: array of H/A/D (other values are ignored). Returns (dead_model, team_model)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    labels = np.asarray(labels)
    known = np.isin(labels, ["H", "A", "D"])
    dead = HistGradientBoostingClassifier(**HGB).fit(
        np.concatenate([X[known], Xs[known]]), np.tile(labels[known] == "D", 2))
    live = np.isin(labels, ["H", "A"])
    team = HistGradientBoostingClassifier(**HGB).fit(
        np.concatenate([X[live], Xs[live]]),
        np.concatenate([labels[live] == "H", labels[live] == "A"]))
    return dead, team


def predict_proba(models, X, Xs):
    """Swap-consistent probabilities, columns H, A, D."""
    dead, team = models
    p_dead = (dead.predict_proba(X)[:, 1] + dead.predict_proba(Xs)[:, 1]) / 2
    p_home = (team.predict_proba(X)[:, 1] + 1 - team.predict_proba(Xs)[:, 1]) / 2
    live = 1 - p_dead
    return np.column_stack((live * p_home, live * (1 - p_home), p_dead))
