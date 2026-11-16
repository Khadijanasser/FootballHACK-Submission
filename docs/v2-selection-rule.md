# V2 selection rule (recorded before any validation scoring)

Recorded: 2026-11-16T01:27:17Z, after development cross-fitting and before the
validation matches (J03WPY, J03WQQ) were scored by any method.

Frozen candidate: V2 gradient-boosted dead/live and home/away models (settings in
`src/footballhack/v2/model.py`), trained on J03WN1, J03WOY and J03WR9 with the seven
training conditions and stride in `experiment.py`, decoded with Viterbi penalties
turnover = stop = restart = 10. The decoder was chosen among the five preregistered
candidates (argmax, 2, 5, 10, 20) using development cross-fit results only.

Rule. V2 replaces V1 as the submission model only if, on the two validation matches,
scored by the unchanged official scorer on identical inputs:

1. V2's equal-match mean spellF1 exceeds V1's by at least 0.04 (the organisers' stated
   noise band) in every one of the four evaluation conditions
   (clean, ball15_structured, ball41_structured, ball41_harsh); and
2. V2's spellF1 is higher than V1's on each validation match individually in the clean
   and ball41_harsh conditions; and
3. V2's equal-match mean macroF1 is not more than 0.01 below V1's in any condition.

Otherwise V1 (frozen causal pipeline) remains the submission. No hyperparameter,
feature, decoder or corruption setting is changed after validation scores are seen.
The sealed holdout (J03WMX, J03WOH) is not used. No competition data or ground truth
takes part in this decision.
