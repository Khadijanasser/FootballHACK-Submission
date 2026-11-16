"""External (Sportec) training and evaluation for V2. No competition data.

Whole-match cross-fitting on development matches, final training on all
development matches, and frozen evaluation on validation matches. The same
corrupted inputs are given to every compared method; the official scorer is
used unchanged, per half, and averaged over halves then matches.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import decode, masks, model
from .data import PARTITIONS, load_half

TRAIN_CONDITIONS = ("clean", "ball10_structured", "ball20_structured", "ball41_structured",
                    "ball41_uniform", "ball41_harsh", "ball60_harsh")
EVAL_CONDITIONS = ("clean", "ball15_structured", "ball41_structured", "ball41_harsh")
STRIDE = 3
# Preregistered decoder set (uniform penalties); "argmax" is the direct frame prediction.
DECODERS = {"argmax": None, "viterbi2": 2.0, "viterbi5": 5.0, "viterbi10": 10.0, "viterbi20": 20.0}
ALL_MATCHES = sorted(PARTITIONS["development"] + PARTITIONS["validation"])


def _seed(match, half, condition, purpose):
    return [ALL_MATCHES.index(match), half, list(masks.CONDITIONS).index(condition), purpose]


def training_rows(cache_dir, matches, *, stride=STRIDE):
    Xs_, Xo_, y_ = [], [], []
    for match in matches:
        for half in (1, 2):
            tracking, labels = load_half(cache_dir, match, half)
            for condition in TRAIN_CONDITIONS:
                players, ball = masks.corrupt(tracking, condition, _seed(match, half, condition, 0))
                X, Xs, names = model.half_features(players, ball, tracking["teams"], tracking["pitch_m"],
                                                   fps=tracking["fps"])
                offset = TRAIN_CONDITIONS.index(condition) % stride
                Xo_.append(X[offset::stride]); Xs_.append(Xs[offset::stride]); y_.append(labels[offset::stride])
    return np.concatenate(Xo_), np.concatenate(Xs_), np.concatenate(y_), names


def _v1_labels(players, ball, tracking):
    """Frozen V1 causal pipeline on exactly the same (possibly corrupted) positions."""
    from footballhack.io.atd import ATDMetadata, TrackSlot, TrackingData
    from footballhack.pipeline import PipelineConfig, predict_possession
    teams = tracking["teams"]
    slots = tuple(TrackSlot(str(i), side, side, str(i)) for i, side in enumerate(teams))
    meta = ATDMetadata(("H", "A"), ("H", "A"), slots, tracking["fps"], tuple(tracking["pitch_m"]), False)
    n = len(ball)
    data = TrackingData(np.arange(n), players, ball, teams, meta)
    from footballhack.io import Period
    return predict_possession(data, config=PipelineConfig(n_frames=n, periods=(Period(1, 0, n),))).labels


def score(reference, predicted, ball_missing=None):
    from footballhack.external._official import possession_metrics as pm
    gt, pred = pm.label_codes(reference), pm.label_codes(predicted)
    r = pm.evaluate(gt, pred)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    g, p = pm.transitions(gt, runs), pm.transitions(filled, runs)
    def typed(select):
        s = pm.transition_scores([t for t in g if select(t)], [t for t in p if select(t)], 25, typed="pair")
        return dict(precision=s["precision"], recall=s["recall"], n_ref=s["n_gt"], n_pred=s["n_pred"],
                    n_spurious=s["n_spurious"])
    out = dict(spell_f1=r["spell_f1_mean"], macro_f1=r["macro_f1"], n_ref_spells=r["n_gt_spells"],
               n_pred_spells=r["n_pred_spells"], per_class=r["per_class"],
               x_rate=float(np.mean(pred[gt != pm.X] == pm.X)),
               turnover=typed(lambda t: {t[1], t[2]} == {pm.H, pm.A}),
               restart=typed(lambda t: t[1] == pm.D and t[2] in (pm.H, pm.A)),
               stoppage=typed(lambda t: t[2] == pm.D))
    if ball_missing is not None and ball_missing.any():
        keep = ball_missing & (gt != pm.X)
        out["accuracy_ball_missing"] = float(np.mean(filled[keep] == gt[keep]))
        out["n_ball_missing"] = int(keep.sum())
    return out


def evaluate_match(models, cache_dir, match, *, save_dir=None, extra=None):
    """Score V1 and every V2 decoder on each eval condition. Returns nested dict."""
    results = {}
    for condition in EVAL_CONDITIONS:
        per_method = {}
        for half in (1, 2):
            tracking, labels = load_half(cache_dir, match, half)
            players, ball = masks.corrupt(tracking, condition, _seed(match, half, condition, 1))
            missing = ~np.isfinite(ball).all(axis=1)
            X, Xs, _ = model.half_features(players, ball, tracking["teams"], tracking["pitch_m"], fps=tracking["fps"])
            proba = model.predict_proba(models, X, Xs)
            v1 = _v1_labels(players, ball, tracking)
            if save_dir is not None:
                np.savez_compressed(Path(save_dir) / f"{match}_h{half}_{condition}.npz",
                                    proba=proba.astype(np.float32), v1=v1, labels=labels, missing=missing)
            predictions = {"v1_causal": v1}
            for name, penalty in DECODERS.items():
                predictions["v2_" + name] = (decode.STATES[np.argmax(proba, axis=1)] if penalty is None else
                                             decode.viterbi(proba, turnover=penalty, stop=penalty, restart=penalty))
            if extra is not None:
                predictions.update(extra(proba, v1))
            for name, labels_pred in predictions.items():
                per_method.setdefault(name, []).append(score(labels, labels_pred, missing))
        results[condition] = per_method
    return results


def summarise(by_match):
    """Equal-half mean per match, then equal-match mean. Returns {condition: {method: {...}}}."""
    table = {}
    for condition in EVAL_CONDITIONS:
        table[condition] = {}
        for method in next(iter(by_match.values()))[condition]:
            rows = {}
            for match, result in by_match.items():
                halves = result[condition][method]
                def pooled(cls, key):
                    return float(np.mean([h["per_class"][cls][key] for h in halves]))
                rows[match] = dict(
                    spell_f1=float(np.mean([h["spell_f1"] for h in halves])),
                    macro_f1=float(np.mean([h["macro_f1"] for h in halves])),
                    d_precision=pooled("D", "precision"), d_recall=pooled("D", "recall"), d_f1=pooled("D", "f1"),
                    n_pred_spells=sum(h["n_pred_spells"] for h in halves),
                    n_ref_spells=sum(h["n_ref_spells"] for h in halves),
                    x_rate=float(np.mean([h["x_rate"] for h in halves])),
                    turnover_recall=float(np.mean([h["turnover"]["recall"] for h in halves])),
                    turnover_precision=float(np.mean([h["turnover"]["precision"] for h in halves])),
                    false_switches=sum(h["turnover"]["n_spurious"] for h in halves),
                    restart_recall=float(np.mean([h["restart"]["recall"] for h in halves])),
                    stoppage_recall=float(np.mean([h["stoppage"]["recall"] for h in halves])),
                    accuracy_ball_missing=float(np.mean([h.get("accuracy_ball_missing", np.nan) for h in halves])))
            mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))}
            table[condition][method] = dict(matches=rows, mean=mean)
    return table


def print_table(table):
    for condition, methods in table.items():
        print(f"\n== {condition} ==")
        print(f"{'method':<16}{'spellF1':>8}{'macroF1':>8}{'D-P':>6}{'D-R':>6}{'spells':>11}{'X%':>6}{'TO-R':>6}{'TO-P':>6}{'RS-R':>6}{'missAcc':>8}  per-match spellF1")
        for method, v in methods.items():
            m = v["mean"]
            per = " ".join(f"{r['spell_f1']:.3f}" for r in v["matches"].values())
            spells = f"{sum(r['n_pred_spells'] for r in v['matches'].values())}/{sum(r['n_ref_spells'] for r in v['matches'].values())}"
            print(f"{method:<16}{m['spell_f1']:>8.4f}{m['macro_f1']:>8.4f}{m['d_precision']:>6.2f}{m['d_recall']:>6.2f}"
                  f"{spells:>11}{100 * m['x_rate']:>6.1f}{m['turnover_recall']:>6.2f}{m['turnover_precision']:>6.2f}"
                  f"{m['restart_recall']:>6.2f}{m['accuracy_ball_missing']:>8.3f}  {per}")


def crossfit(cache_dir, out_dir):
    matches = PARTITIONS["development"]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for match in matches:
        start = time.time()
        rows[match] = training_rows(cache_dir, [match])
        print(f"features {match}: {rows[match][0].shape} in {time.time() - start:.0f}s", flush=True)
    by_match = {}
    for held in matches:
        start = time.time()
        train = [m for m in matches if m != held]
        X = np.concatenate([rows[m][0] for m in train]); Xs = np.concatenate([rows[m][1] for m in train])
        y = np.concatenate([rows[m][2] for m in train])
        models = model.fit(X, Xs, y)
        del X, Xs
        print(f"fold {held}: trained in {time.time() - start:.0f}s", flush=True)
        by_match[held] = evaluate_match(models, cache_dir, held, save_dir=out_dir)
        print(f"fold {held}: evaluated at {time.time() - start:.0f}s", flush=True)
    table = summarise(by_match)
    (out_dir / "crossfit-summary.json").write_text(json.dumps(table, indent=1))
    print_table(table)


def final_train(cache_dir, model_path):
    import joblib
    X, Xs, y, names = training_rows(cache_dir, PARTITIONS["development"])
    models = model.fit(X, Xs, y)
    joblib.dump(dict(models=models, feature_names=names, hgb=model.HGB,
                     train_matches=PARTITIONS["development"], train_conditions=TRAIN_CONDITIONS, stride=STRIDE),
                model_path, compress=3)
    print("saved", model_path, len(names), "features", len(y), "rows")


def validate(cache_dir, model_path, out_dir):
    import joblib
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models = joblib.load(model_path)["models"]
    by_match = {m: evaluate_match(models, cache_dir, m, save_dir=out_dir) for m in PARTITIONS["validation"]}
    table = summarise(by_match)
    (out_dir / "validation-summary.json").write_text(json.dumps(table, indent=1))
    print_table(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("crossfit", "final-train", "validate"))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.command == "crossfit":
        crossfit(args.cache_dir, args.out_dir)
    elif args.command == "final-train":
        final_train(args.cache_dir, args.model)
    else:
        validate(args.cache_dir, args.model, args.out_dir)
