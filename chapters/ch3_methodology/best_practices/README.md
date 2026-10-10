# Best practices: protocol comparison

Answers part of RQ1: whether a single train/test split gives a trustworthy accuracy estimate, relative to an
expanding-window backtest, across five simple baselines (seasonal-naive, naive, random forest, CatBoost, linear
regression).

| File | Produces |
|---|---|
| `protocol.py` | Single source of truth: the two protocol variants (`SPEC_SINGLE_SPLIT`, `SPEC_EXPANDING`), datasets, `base_config()`. See its own docstring for how the expanding protocol's month-equivalents were pinned. |
| `conditions.py` | The five baseline `Condition`s (HPO disabled; the question is the protocol, not model tuning). |
| `experiment.py` | Runs both protocol variants via `ExperimentEngine`. |
| `analysis.py` | Protocol-agreement table (model rank under each protocol), the five-metric table (full-sample; extreme-scenario-only when raw predictions are available), DM + GW significance. |
| `figures.py` | Model x protocol grouped bar; test-time-vs-NRMSE efficiency scatter. |

```bash
uv run python chapters/ch3_methodology/best_practices/experiment.py --protocol all
uv run python chapters/ch3_methodology/best_practices/analysis.py
uv run python -c "from chapters.ch3_methodology.best_practices.figures import render_all; render_all()"
```

Note: MSTL (used by the legacy single-dataset notebook this experiment generalises) has no twiga equivalent and is
dropped rather than substituted.
