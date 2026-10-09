r"""Decision sweep read by forecast type: what each head's output and each scenario form give an operator.

Reads the ``{head}_none`` conditions of ``fixed_decision_sweep.py`` (conformal conditions are ignored) and reports,
following ``FORECAST_TYPE_PLAN.md``:

* **event rates** per scenario, and which scenario-folds can be scored: an event must occur on 2-98% of the fold's
  test steps or days, so that a probability forecast has something to discriminate;
* **single-step events** (a half-hour above the thermal limit, a customer-voltage breach): which head gives the
  best probabilities, in its primary form;
* **sustained and combined events** (an overload lasting two hours, an overload or voltage breach on the same
  day): what each scenario form does -- independent draws, Schaake coupling with recent days, and the BNN's own
  paths with per-cell or day-coherent noise -- as mean forecast probability against the realised rate and as Brier
  score;
* **significance**, as everywhere in the thesis: the Diebold-Mariano test on the daily Brier losses each condition
  saves (one per forecast day, form and event), per scenario, with Benjamini-Hochberg adjustment across the
  scenarios of each comparison; the summary counts the scenarios where each form is significantly better;
* the copula's effect across the EV x PV grid (median Brier change of Schaake against independent draws).

The primary form is the one each head would be used with: Schaake coupling for the quantile and Laplace heads,
native paths for the BNN.

Usage::

    uv run python experiment/ch7_multitarget/experiment_3_decision/decision_forecast_type_report.py \
        --experiment-root <sweep root>/fixed_benchmark/multitarget_fixed_decision_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(script_dir), str(script_dir.parents[1])]

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from twiga_thesis.analysis import dm_and_mcs  # noqa: E402
from twiga_thesis.plotting import line_style as LS, style as S, tokens as T  # noqa: E402

HEADS = {"fpqr_anchor": "FPQR", "qr": "QR", "laplace": "Laplace", "bnn": "BNN"}
HEAD_TEX = {"fpqr_anchor": r"\ac{FPQR}", "qr": r"\ac{QR}", "laplace": "Laplace", "bnn": r"\ac{BNN}"}
# The benchmark analysis's head colours (fixed_benchmark_chapter_figures.COLOUR), so a head reads the same everywhere.
HEAD_COLOUR = {"fpqr_anchor": T.PALETTE[2], "qr": T.PALETTE[0], "bnn": T.PALETTE[4], "laplace": T.PALETTE[5]}
REFERENCE = "fpqr_anchor"
NETWORK_TEX = {"DSS-UK:weak": r"\ac{DSS-UK} weak", "DSS-UK:stiff": r"\ac{DSS-UK} stiff", "DSS-US:8": r"\ac{DSS-US}"}
NETWORK_SLUG = {"DSS-UK:weak": "dssuk_weak", "DSS-UK:stiff": "dssuk_stiff", "DSS-US:8": "dssus"}
EVENT_TEX = {
    "overload_1x_step": r"$1.0\times$",
    "overload_1.25x_step": r"$1.25\times$",
    "overload_1.5x_step": r"$1.5\times$",
    "voltage_step": r"\ac{V}",
    "overload_1x_day": r"$1.0\times$",
    "overload_1.25x_day": r"$1.25\times$",
    "overload_1.5x_day": r"$1.5\times$",
    "union_day": r"Union",
}
IMAGES = script_dir.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
FORMS = {"independent": "independent", "schaake": "Schaake", "native": "native", "native_day": "native, day noise"}
PRIMARY = {"fpqr_anchor": "schaake", "qr": "schaake", "laplace": "schaake", "bnn": "native"}
STEP_EVENTS = ("overload_1x_step", "overload_1.25x_step", "overload_1.5x_step", "voltage_step")
DAY_EVENTS = ("overload_1x_day", "overload_1.25x_day", "overload_1.5x_day", "union_day")
USABLE = (0.02, 0.98)
MIN_DAYS = 20
FORM_PAIRS = (("schaake", "independent"), ("native", "schaake"), ("native_day", "native"), ("native_day", "schaake"))
# Brier change: teal where coupling lowers it (better), orange where it raises it, white at none.
CHANGE_CMAP = LinearSegmentedColormap.from_list("thesis_change", [T.PRIMARY, T.WHITE, T.DANGER])


def scenario_key(run: Path) -> str | None:
    """The run's ``record:feeder:ev:pv`` key, from ``meta.json``, else ``run_status.json``, else the folder name.

    Interrupted runs can lack both files; their condition folder (``dss-uk_weak_0_2_0`` is EV 0.2, PV 0) still
    identifies the scenario.
    """
    meta = run / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text())["datasets"][0]
    status = run / "run_status.json"
    if status.exists():
        conditions = json.loads(status.read_text()).get("conditions", [])
        if conditions:
            return conditions[0]["dataset"]
    folders = sorted((run / "conditions").glob("*"))
    if not folders:
        return None
    record, feeder, *tokens = folders[0].name.split("_")
    values, i = [], 0
    while i < len(tokens):
        if tokens[i] == "0" and i + 1 < len(tokens) and tokens[i + 1] != "0":
            values.append(f"0.{tokens[i + 1]}")
            i += 2
        else:
            values.append(tokens[i])
            i += 1
    if record == "dss-us":  # the US key holds the transformer rank in the feeder slot
        return f"DSS-US:{feeder}:{values[0]}:{values[1]}"
    return f"{record.upper()}:{feeder}:{values[0]}:{values[1]}"


def newest_condition_files(experiment_root: Path, filename: str) -> list[tuple[str, Path]]:
    """``(scenario key, file)`` for each scenario and head, from the newest run in which that head finished.

    Heads are taken run by run rather than whole runs, because a re-score that had to retrain some heads
    (``fixed_decision_sweep.py --rescore-from``) leaves them in a run of their own beside the reused ones.
    """
    found, seen = [], set()
    for run in sorted((experiment_root / "runs").glob("*"), reverse=True):
        key = scenario_key(run)
        if key is None:
            continue
        for path in run.glob(f"conditions/*/decision/*_none/{filename}"):
            head = path.parent.name.removesuffix("_none")
            if (key, head) in seen or not (path.parent / "metrics.parquet").exists():
                continue
            seen.add((key, head))
            found.append((key, path))
    return found


def load(experiment_root: Path) -> pd.DataFrame:
    """Every ``{head}_none`` metric row, newest finished run per scenario and head, with network, EV and PV."""
    frames = []
    for key, path in newest_condition_files(experiment_root, "metrics.parquet"):
        record, feeder, ev, pv = key.split(":")
        frame = pd.read_parquet(path)
        frames.append(
            frame.assign(
                head=frame["condition"].str.removesuffix("_none"),
                dependence=frame["dependence"].fillna("native"),
                network=f"{record}:{feeder}",
                ev=float(ev),
                pv=float(pv),
            )
        )
    if not frames:
        raise SystemExit(f"no decision runs under {experiment_root}")
    return pd.concat(frames, ignore_index=True)


def probability_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Probability-tier metrics, one row per (scenario, head, form, event, fold), with the realised rate."""
    keys = ["network", "ev", "pv", "head", "dependence", "event", "Folds"]
    prob = rows[rows["tier"] == "probability"].pivot_table(index=keys, columns="metric", values="value")
    rates = rows[rows["tier"] == "validity"].pivot_table(
        index=["network", "ev", "pv", "event", "Folds"], columns="metric", values="value"
    )
    return prob.reset_index().merge(rates.reset_index(), on=["network", "ev", "pv", "event", "Folds"])


def usable(prob: pd.DataFrame) -> pd.DataFrame:
    """Scenario-folds whose event occurs on 2-98% of test steps or days, with a non-zero reference rate."""
    low, high = USABLE
    return prob[(prob["realised_rate"] > low) & (prob["realised_rate"] < high) & (prob["reference_rate"] > 0)]


def load_daily(experiment_root: Path) -> pd.DataFrame:
    """Daily Brier losses of every ``{head}_none`` condition (as :func:`load`); empty where runs predate them."""
    frames = []
    for key, path in newest_condition_files(experiment_root, "predictions.parquet"):
        frame = pd.read_parquet(path)
        if "brier" not in frame.columns:
            continue
        record, feeder, ev, pv = key.split(":")
        head = path.parent.name.removesuffix("_none")
        window = frame["Folds"].astype(str) + "-" + pd.to_datetime(frame["forecast_origin"]).astype(str)
        frames.append(frame.assign(head=head, network=f"{record}:{feeder}", ev=float(ev), pv=float(pv), window=window))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values."""
    order = p.sort_values()
    m = len(order)
    adjusted = (order * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    return adjusted.reindex(p.index)


def dm_by_scenario(daily: pd.DataFrame, scorable: pd.Index, alpha: float = 0.05) -> pd.DataFrame:
    """DM test of each pair of scenario forms per (scenario, head, event), on the daily Brier losses.

    Only scenarios where the event is scorable (``scorable``: network, EV, PV, event) are tested; p-values are
    adjusted across the scenarios of each (head, event, comparison).
    """
    rows = []
    keys = ["network", "ev", "pv", "head", "event"]
    for (network, ev, pv, head, event), g in daily.groupby(keys):
        if (network, ev, pv, event) not in scorable:
            continue
        for a, b in FORM_PAIRS:
            pair = g[g["dependence"].isin([a, b])]
            if pair["dependence"].nunique() < 2 or pair["window"].nunique() < MIN_DAYS:
                continue
            table, _, _ = dm_and_mcs(pair, metric="brier", fold_column="window", condition_column="dependence")
            r = table.iloc[0]
            mean = pair.groupby("dependence")["brier"].mean()
            rows.append(
                {
                    "network": network,
                    "ev": ev,
                    "pv": pv,
                    "head": head,
                    "event": event,
                    "comparison": f"{a} vs {b}",
                    "a_mean": mean[a],
                    "b_mean": mean[b],
                    "p_value": r.p_value,
                    "winner": r.winner,
                }
            )
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests
    tests["p_adjusted"] = tests.groupby(["head", "event", "comparison"])["p_value"].transform(benjamini_hochberg)
    tests["significant"] = tests["p_adjusted"] < alpha
    return tests


def dm_summary(tests: pd.DataFrame) -> pd.DataFrame:
    """Per (event, comparison, head): scenarios tested, and where the first or second form is significantly better."""
    if tests.empty:
        return tests
    first = tests["comparison"].str.split(" vs ").str[0]
    tests = tests.assign(
        a_better=tests["significant"] & (tests["winner"] == first),
        b_better=tests["significant"] & (tests["winner"] != first),
    )
    return (
        tests.groupby(["event", "comparison", "head"])
        .agg(scenarios=("p_value", "size"), a_better=("a_better", "sum"), b_better=("b_better", "sum"))
        .reset_index()
    )


def tables(ok: pd.DataFrame, daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The report's tables: head ranking on step events, forms on day events, and the DM tests."""
    primary = ok[ok["dependence"] == ok["head"].map(PRIMARY)]
    step = primary[primary["event"].isin(STEP_EVENTS)]
    out = {
        "step_events": step.groupby(["event", "head"])[["brier_score", "brier_skill_score", "reliability"]]
        .mean()
        .reset_index(),
        "forms": ok.groupby(["event", "head", "dependence"])[
            ["brier_score", "mean_probability", "realised_rate", "reliability", "resolution"]
        ]
        .mean()
        .reset_index(),
    }
    if daily.empty:
        return out
    rates = ok.groupby(["network", "ev", "pv", "event"])["realised_rate"].mean()
    scorable = rates[(rates > USABLE[0]) & (rates < USABLE[1])].index
    tests = dm_by_scenario(daily, scorable)
    out |= {"dm_tests": tests, "dm_summary": dm_summary(tests)}
    return out


def _paired(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """One Brier loss per condition and scored scenario-fold: the paired observations the DM test and MCS use."""
    obs = (
        frame["network"]
        + "|"
        + frame["ev"].astype(str)
        + "|"
        + frame["pv"].astype(str)
        + "|"
        + frame["Folds"].astype(str)
    )
    return frame.assign(Folds=obs, condition=frame[by])[["Folds", "condition", "brier_score"]]


def head_significance(ok: pd.DataFrame) -> pd.DataFrame:
    """Each head in its primary form, per network and event: mean Brier and skill, MCS membership, DM against FPQR.

    The observations are the scored (scenario, fold) pairs of the network, one fold-level Brier loss per head, so the
    test is the benchmark's fold-level DM and MCS (``dm_and_mcs``) with the penetration scenarios as extra folds.
    """
    primary = ok[ok["dependence"] == ok["head"].map(PRIMARY)]
    rows = []
    for (network, event), g in primary.groupby(["network", "event"]):
        pairs = _paired(g, "head").pivot_table(index="Folds", columns="condition", values="brier_score").dropna()
        heads = [h for h in HEADS if h in pairs.columns]
        if len(heads) < 2 or len(pairs) < 3:
            continue
        long = pairs[heads].reset_index().melt(id_vars="Folds", var_name="condition", value_name="brier_score")
        dm, _, model_set = dm_and_mcs(long, metric="brier_score")
        included = set(map(str, model_set.included))
        kept = g[_paired(g, "head")["Folds"].isin(pairs.index)]
        for head in heads:
            hg = kept[kept["head"] == head]
            arrow = None
            for r in dm.itertuples(index=False):
                if {r.condition_a, r.condition_b} == {head, REFERENCE} and head != REFERENCE and r.significant:
                    arrow = "up" if r.winner == head else "down"
            rows.append(
                {
                    "network": network,
                    "event": event,
                    "head": head,
                    "observations": len(pairs),
                    "brier_score": hg["brier_score"].mean(),
                    "brier_skill_score": hg["brier_skill_score"].mean(),
                    "reliability": hg["reliability"].mean(),
                    "in_mcs": head in included,
                    "versus_fpqr": arrow,
                }
            )
    return pd.DataFrame(rows)


def form_significance(ok: pd.DataFrame) -> pd.DataFrame:
    """Each head's scenario forms on the day events, pooled over networks.

    Mean Brier score, MCS membership within the head, and the DM direction against independent draws.
    """
    rows = []
    day = ok[ok["event"].isin(DAY_EVENTS)]
    # A union day equals the sustained overload where no voltage event can occur: keep it only where one can.
    with_voltage = set(ok.loc[ok["event"] == "voltage_step", "network"])
    day = day[(day["event"] != "union_day") | day["network"].isin(with_voltage)]
    for (head, event), g in day.groupby(["head", "event"]):
        pairs = _paired(g, "dependence").pivot_table(index="Folds", columns="condition", values="brier_score").dropna()
        forms = [f for f in FORMS if f in pairs.columns]
        if len(forms) < 2 or len(pairs) < 3:
            continue
        long = pairs[forms].reset_index().melt(id_vars="Folds", var_name="condition", value_name="brier_score")
        dm, _, model_set = dm_and_mcs(long, metric="brier_score")
        included = set(map(str, model_set.included))
        for form in forms:
            arrow = None
            for r in dm.itertuples(index=False):
                if {r.condition_a, r.condition_b} == {form, "independent"} and form != "independent" and r.significant:
                    arrow = "up" if r.winner == form else "down"
            fg = g[(g["dependence"] == form)]
            rows.append(
                {
                    "head": head,
                    "event": event,
                    "form": form,
                    "observations": len(pairs),
                    "brier_score": pairs[form].mean(),
                    "mean_probability": fg["mean_probability"].mean(),
                    "realised_rate": fg["realised_rate"].mean(),
                    "in_mcs": form in included,
                    "versus_independent": arrow,
                }
            )
    return pd.DataFrame(rows)


def _cell(value: float, digits: int, bold: bool, arrow: str | None = None) -> str:
    """A table cell in the benchmark tables' convention: bold for the MCS, an arrow for a significant DM test."""
    if not np.isfinite(value):
        return "--"
    text = f"{value:.{digits}f}"
    text = rf"\B{text}" if bold else text
    if arrow in ("up", "down"):  # missing tests arrive as NaN, which is truthy
        text += r"$^{\uparrow}$" if arrow == "up" else r"$^{\downarrow}$"
    return text


def decision_table(heads: pd.DataFrame) -> str:
    """Brier skill of each head in its primary form, per network and event (tab:ch6-decision)."""
    events = [e for e in (*STEP_EVENTS, *DAY_EVENTS) if e in set(heads["event"])]
    steps = [e for e in events if e in STEP_EVENTS]
    days = [e for e in events if e in DAY_EVENTS]
    caption = (
        r"Brier skill of each head's event probabilities in its primary form (Schaake-coupled draws for "
        r"\ac{FPQR}, \ac{QR} and Laplace, native paths for the \ac{BNN}), averaged over the scenario-folds of each "
        r"network in which the event occurs on 2--98\% of test steps or days. Overload tiers are multiples of the "
        r"network's pre-\ac{DER} thermal limit; day events require at least two hours above it; \ac{V} is a "
        r"customer-voltage breach and Union a day with either event (\ac{DSS-UK} only). Higher is better. Bold marks "
        r"membership of the 95\% \acl{MCS} and $^{\uparrow}$/$^{\downarrow}$ a head significantly better/worse than "
        r"\ac{FPQR} (\ac{DM} test on paired scenario-fold Brier scores, Benjamini-Hochberg adjusted, 5\%)."
    )
    lines = [
        r"\begin{table*}[ht]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        r"    \begin{adjustbox}{width=\textwidth,center}",
        r"    \begin{tabular}{ll" + "c" * len(events) + r"} \toprule",
        rf"    & & \multicolumn{{{len(steps)}}}{{c}}{{Single step}} "
        rf"& \multicolumn{{{len(days)}}}{{c}}{{Sustained (day)}} \\",
        rf"    \cmidrule(lr){{3-{2 + len(steps)}}} \cmidrule(lr){{{3 + len(steps)}-{2 + len(events)}}}",
        r"    Network & Model & " + " & ".join(EVENT_TEX[e] for e in events) + r" \\",
        r"    \midrule",
    ]
    networks = [n for n in NETWORK_TEX if n in set(heads["network"])]
    for k, network in enumerate(networks):
        for i, head in enumerate(HEADS):
            cells = []
            for event in events:
                row = heads[(heads["network"] == network) & (heads["event"] == event) & (heads["head"] == head)]
                # A union day equals the sustained overload where no voltage event can occur.
                no_voltage = heads[(heads["network"] == network) & (heads["event"] == "voltage_step")].empty
                if row.empty or (event == "union_day" and no_voltage):
                    cells.append("--")
                    continue
                r = row.iloc[0]
                cells.append(_cell(r["brier_skill_score"], 3, bool(r["in_mcs"]), r["versus_fpqr"]))
            first = rf"\multirow{{{len(HEADS)}}}{{*}}{{{NETWORK_TEX[network]}}}" if i == 0 else ""
            lines.append(f"    {first} & {HEAD_TEX[head]} & " + " & ".join(cells) + r" \\")
        if k < len(networks) - 1:
            lines.append(r"    \midrule")
    lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"    \end{adjustbox}",
        r"    \label{tab:ch6-decision}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def forms_table(forms: pd.DataFrame) -> str:
    """Brier score of each scenario form on the sustained events, per head (tab:ch6-decision-forms)."""
    events = [e for e in DAY_EVENTS if e in set(forms["event"])]
    caption = (
        r"Brier score of the sustained-overload and union events by scenario form, pooled over the scored "
        r"scenario-folds of all networks. Independent draws ignore dependence; Schaake draws reorder the same "
        r"marginal values by the ranks of recent observed days; the \ac{BNN}'s native paths add per-step noise to "
        r"each weight draw, and its day-noise variant holds the noise over the day. Lower is better. Bold marks the "
        r"95\% \acl{MCS} within each head and $^{\uparrow}$/$^{\downarrow}$ a form significantly better/worse than "
        r"independent draws (\ac{DM} test, Benjamini-Hochberg adjusted, 5\%)."
    )
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        r"    \begin{tabular}{ll" + "c" * len(events) + r"} \toprule",
        r"    Model & Form & " + " & ".join(EVENT_TEX[e] for e in events) + r" \\",
        r"    \midrule",
    ]
    for k, head in enumerate(HEADS):
        head_forms = [f for f in FORMS if not forms[(forms["head"] == head) & (forms["form"] == f)].empty]
        for i, form in enumerate(head_forms):
            cells = []
            for event in events:
                row = forms[(forms["head"] == head) & (forms["event"] == event) & (forms["form"] == form)]
                if row.empty:
                    cells.append("--")
                    continue
                r = row.iloc[0]
                cells.append(_cell(r["brier_score"], 3, bool(r["in_mcs"]), r["versus_independent"]))
            first = rf"\multirow{{{len(head_forms)}}}{{*}}{{{HEAD_TEX[head]}}}" if i == 0 else ""
            lines.append(f"    {first} & {FORMS[form].capitalize()} & " + " & ".join(cells) + r" \\")
        if k < len(HEADS) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"    \end{tabular}", r"    \label{tab:ch6-decision-forms}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _scored_keys(ok: pd.DataFrame) -> pd.DataFrame:
    return ok[["network", "ev", "pv", "event", "Folds"]].drop_duplicates()


def value_frame(rows: pd.DataFrame, ok: pd.DataFrame) -> pd.DataFrame:
    """Relative economic value of each head's warnings in its primary form, on the scored scenario-folds.

    ``value_curve`` rows give REV over the full cost-loss range, ``warning`` rows at the three operating points.
    """
    keys = _scored_keys(ok)
    frames = []
    for tier in ("value_curve", "warning"):
        v = rows[(rows["tier"] == tier) & (rows["metric"] == "relative_value")]
        v = v[v["dependence"].fillna("native") == v["head"].map(PRIMARY)]
        frames.append(v.merge(keys, on=["network", "ev", "pv", "event", "Folds"]).assign(source=tier))
    return pd.concat(frames, ignore_index=True)


def plot_event_rates(rates: pd.DataFrame, out: Path) -> list[Path]:
    """Realised rate of the sustained 1.0x overload against EV penetration, one line per PV level, per network."""
    paths = []
    event = "overload_1x_day"
    # One thesis colour and marker per PV level, so the lines stay distinct in greyscale; PV 0 is the grey reference.
    styles = {
        0.0: (T.GRAY_600, "o", "--"),
        0.2: (T.INFO, "s", "-"),
        0.4: (T.PRIMARY, "^", "-"),
        0.6: (T.HIGHLIGHT, "D", "-"),
    }
    for network, slug in NETWORK_SLUG.items():
        if network not in rates.index.get_level_values("network") or event not in rates.columns:
            continue
        grid = rates.loc[network, event].unstack("pv")
        fig, ax = LS.new_panel(0.31, 0.9)
        for pv in grid.columns:
            colour, marker, linestyle = styles.get(pv, (T.PRIMARY, "o", "-"))
            ax.plot(
                100 * grid.index,
                grid[pv],
                marker=marker,
                ls=linestyle,
                color=colour,
                ms=3.5,
                mfc="white",
                mew=LS.LINE_WIDTH,
                lw=LS.LINE_WIDTH,
                label=rf"PV {100 * pv:.0f}\%",
            )
        LS.finish(ax)
        ax.set_xticks([0, 20, 40, 60])
        LS.set_y_axis(ax, unit_interval=True)
        ax.set_xlabel(r"EV penetration (\%)")
        ax.set_ylabel("Event rate")
        LS.legend_below(ax, ncol=4)
        path = out / f"fig_ch6_decision_event_rate_{slug}"
        LS.save(fig, path)
        paths.append(path.with_suffix(".pdf"))
    return paths


def plot_probability_vs_rate(forms: pd.DataFrame, out: Path) -> list[Path]:
    """Mean forecast probability per head and form against the realised rate, one panel per sustained event."""
    colours = {"independent": T.GRAY_400, "schaake": T.PRIMARY, "native": T.HIGHLIGHT, "native_day": T.WARNING}
    paths = []
    for event in [e for e in DAY_EVENTS if e in set(forms["event"])]:
        g = forms[forms["event"] == event]
        fig, ax = LS.new_panel(0.48, 0.62)
        x = np.arange(len(HEADS))
        width = 0.8 / len(FORMS)
        for k, form in enumerate(FORMS):
            values = [g[(g["head"] == h) & (g["dependence"] == form)]["mean_probability"].mean() for h in HEADS]
            ax.bar(
                x + (k - (len(FORMS) - 1) / 2) * width,
                values,
                width,
                color=colours[form],
                label=FORMS[form].capitalize(),
                lw=0,
            )
        LS.reference_line(ax, float(g["realised_rate"].mean()), label="Realised rate")
        ax.set_xticks(x, list(HEADS.values()))
        LS.finish(ax)
        LS.set_y_axis(ax, unit_interval=True)
        ax.set_ylabel("Probability")
        LS.legend_below(ax, ncol=3)
        path = out / f"fig_ch6_decision_probability_{event.replace('.', 'p')}"
        LS.save(fig, path)
        paths.append(path.with_suffix(".pdf"))
    return paths


def plot_copula_grid(ok: pd.DataFrame, out: Path) -> list[Path]:
    """Median Brier change (%) of Schaake against independent draws on the day events, over EV x PV, per network."""
    pairs = ok[ok["dependence"].isin(["schaake", "independent"]) & ok["event"].isin(DAY_EVENTS)]
    wide = pairs.pivot_table(
        index=["network", "ev", "pv", "head", "event", "Folds"], columns="dependence", values="brier_score"
    ).dropna()
    if wide.empty or not {"schaake", "independent"} <= set(wide.columns):
        return []
    change = (100 * (wide["schaake"] / wide["independent"] - 1)).groupby(level=["network", "ev", "pv"]).median()
    limit = float(np.nanmax(np.abs(change)))
    paths = []
    for network, slug in NETWORK_SLUG.items():
        if network not in change.index.get_level_values("network"):
            continue
        grid = change.loc[network].unstack("pv")
        fig, ax = LS.new_panel(0.31, 0.95)
        ax.imshow(grid.to_numpy(), cmap=CHANGE_CMAP, vmin=-limit, vmax=limit, origin="lower")
        ax.set_xticks(range(grid.shape[1]), [f"{100 * v:.0f}" for v in grid.columns])
        ax.set_yticks(range(grid.shape[0]), [f"{100 * v:.0f}" for v in grid.index])
        for (i, j), value in np.ndenumerate(grid.to_numpy()):
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.0f}", ha="center", va="center", fontsize=7)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel(r"PV penetration (\%)")
        ax.set_ylabel(r"EV penetration (\%)")
        path = out / f"fig_ch6_decision_copula_{slug}"
        LS.save(fig, path)
        paths.append(path.with_suffix(".pdf"))
    return paths


def plot_value_curves(values: pd.DataFrame, out: Path) -> list[Path]:
    """REV of each head's warnings against the cost-loss ratio, pooled over networks, one panel per event."""
    curve = values[values["source"] == "value_curve"]
    paths = []
    for event in ("overload_1x_step", "overload_1x_day"):
        g = curve[curve["event"] == event]
        if g.empty:
            continue
        fig, ax = LS.new_panel(0.48, 0.62)
        LS.reference_line(ax, 0.0)
        for head, label in HEADS.items():
            m = g[g["head"] == head].groupby("cost_loss")["value"].mean()
            ax.plot(m.index, m.to_numpy(), color=HEAD_COLOUR[head], lw=LS.LINE_WIDTH * 1.3, label=label)
        LS.finish(ax)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylim(-0.25, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(r"Cost-loss ratio $C/L$")
        ax.set_ylabel("REV")
        LS.legend_below(ax, ncol=4)
        path = out / f"fig_ch6_decision_value_{event.replace('.', 'p')}"
        LS.save(fig, path)
        paths.append(path.with_suffix(".pdf"))
    return paths


def main() -> None:
    """Build the decision report: tables and CSVs under analysis/decision, figures under analysis/chapter_figures."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment-root", type=Path, required=True, help="the sweep's experiment root (holds runs/)")
    ap.add_argument("--no-latex", action="store_true")
    ap.add_argument("--copy-to-images", action="store_true", help="also copy the figures into Images/Chapter-6")
    args = ap.parse_args()

    LS.configure(latex=not args.no_latex)
    out = args.experiment_root / "analysis" / "decision"
    figures = args.experiment_root / "analysis" / "chapter_figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rows = load(args.experiment_root)
    prob = probability_frame(rows)
    rates = (
        prob.drop_duplicates(["network", "ev", "pv", "event", "Folds"])
        .groupby(["network", "ev", "pv", "event"])["realised_rate"]
        .mean()
        .unstack("event")
    )
    ok = usable(prob)
    coverage = (
        ok.drop_duplicates(["network", "ev", "pv", "event", "Folds"]).groupby("event").size()
        / prob.drop_duplicates(["network", "ev", "pv", "event", "Folds"]).groupby("event").size()
    )
    heads = head_significance(ok)
    forms = form_significance(ok)
    values = value_frame(rows, ok)
    rev = (
        values[values["source"] == "warning"]
        .groupby(["network", "event", "head", "cost_loss"])["value"]
        .mean()
        .unstack("cost_loss")
    )

    # Objective by objective (Chapter 6, Experiment 2): every number behind the tables and figures as CSV.
    rates.to_csv(out / "ch6_decision_event_rates.csv")
    coverage.rename("scored_share").to_csv(out / "ch6_decision_scored_share.csv")
    heads.to_csv(out / "ch6_decision_numbers.csv", index=False)
    forms.to_csv(out / "ch6_decision_forms_numbers.csv", index=False)
    rev.to_csv(out / "ch6_decision_rev.csv")
    # Per-day DM within each scenario (Benjamini-Hochberg across scenarios), where the runs saved daily losses.
    daily = load_daily(args.experiment_root)
    day_tests = pd.DataFrame()
    if not daily.empty:
        scorable = rates.stack().loc[lambda s: (s > USABLE[0]) & (s < USABLE[1])].index
        day_tests = dm_by_scenario(daily, scorable)
        day_tests.to_csv(out / "ch6_decision_dm_daily.csv", index=False)
        dm_summary(day_tests).to_csv(out / "ch6_decision_dm_daily_summary.csv", index=False)
    (out / "ch6_decision.tex").write_text(decision_table(heads))
    (out / "ch6_decision_forms.tex").write_text(forms_table(forms))
    made = plot_event_rates(rates, figures)
    made += plot_probability_vs_rate(
        ok.groupby(["event", "head", "dependence"])[["mean_probability", "realised_rate"]].mean().reset_index(),
        figures,
    )
    made += plot_copula_grid(ok, figures)
    made += plot_value_curves(values, figures)
    if args.copy_to_images and IMAGES.exists():
        for path in made:
            shutil.copy(path, IMAGES / path.name)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print("== 1. Event rates (fold means)\n" + rates.round(3).to_string())
    print("\n== Share of scenario-folds that can be scored\n" + coverage.round(2).to_string())
    print("\n== 2. Heads in their primary form (MCS bold, DM vs FPQR)")
    print(heads.round(3).to_string(index=False))
    print("\n== 3-4. Scenario forms on the day events (MCS within head, DM vs independent)")
    print(forms.round(3).to_string(index=False))
    print("\n== 5. REV at the operating cost-loss ratios (primary form)")
    print(rev.round(3).to_string())
    if day_tests.empty:
        print("\n== Per-day DM: no daily Brier losses in these runs (re-score with --rescore-from)")
    else:
        covered = daily.drop_duplicates(["network", "ev", "pv"]).shape[0]
        print(f"\n== Per-day DM within scenarios ({covered} scenarios with daily losses)")
        print(dm_summary(day_tests).to_string(index=False))
    print(f"\nwritten under {out} and {figures}")


if __name__ == "__main__":
    main()
