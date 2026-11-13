"""Markdown rendering of the result dictionaries built in :mod:`shiftscan.scan`."""

from __future__ import annotations

BASELINE_LABELS = {
    "classifier_auc": "classifier held-out AUC",
    "classifier_accuracy": "classifier held-out accuracy",
    "mmd2": "MMD² (unbiased)",
    "unigram_jsd": "unigram Jensen-Shannon divergence (bits)",
    "length_ks_chars": "length in characters, KS D",
    "length_ks_tokens": "length in tokens, KS D",
    "label_total_variation": "label shares, total variation",
}


def num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def cell(text: str) -> str:
    """Make a feature name safe inside a Markdown table cell."""
    return "`" + text.replace("|", "\\|").replace("`", "'").replace("\n", " ") + "`"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_tests(tests: dict) -> list[str]:
    classifier, mmd = tests["classifier"], tests["mmd"]
    permutation_p = classifier["permutation_p"]
    rows = [
        [
            f"classifier, {classifier['folds']}-fold held-out",
            f"{classifier['n_a']}",
            f"AUC {classifier['auc']:.3f}, accuracy {classifier['accuracy']:.3f}",
            "not run"
            if permutation_p is None
            else f"{num(permutation_p)} ({classifier['permutations']} permutations)",
            f"binomial p {num(classifier['binomial_p'])}",
        ],
        [
            f"MMD, {mmd['kernel']} kernel on TF-IDF",
            f"{mmd['n_a']}",
            f"MMD² {num(mmd['mmd2'])}",
            "not run"
            if mmd["p_value"] is None
            else f"{num(mmd['p_value'])} ({mmd['permutations']} permutations)",
            "null mean " + num(mmd["null_mean"]) + ", null 95 % " + num(mmd["null_q95"]),
        ],
    ]
    lines = ["## Two-sample tests", ""]
    lines += table(["test", "records per side", "effect size", "permutation p-value", "also"], rows)
    return lines


def render_baseline(baseline: dict, only: tuple[str, ...] | None = None) -> list[str]:
    lines = [
        "## Against the split-half baseline",
        "",
        f"{baseline['records_per_part']} records per part, mean of {baseline['repeats']} "
        "random splits. `A vs B` compares a part of A with a part of B; the two `halves` "
        "columns compare two parts of one dataset; `baseline` is their mean.",
        "",
    ]
    rows = []
    for name, values in baseline["statistics"].items():
        if only is not None and name not in only:
            continue
        rows.append(
            [
                BASELINE_LABELS.get(name, name),
                num(values["cross"]),
                num(values["within_a"]),
                num(values["within_b"]),
                num(values["baseline"]),
                num(values["excess"]),
                num(values["ratio"], 3),
            ]
        )
    header = ["statistic", "A vs B", "halves of A", "halves of B", "baseline", "excess", "ratio"]
    lines += table(header, rows)
    if baseline.get("mmd_records_per_side") and (only is None or "mmd2" in only):
        lines += ["", f"MMD² rows use {baseline['mmd_records_per_side']} records per side."]
    return lines


def render_lengths(lengths: dict) -> list[str]:
    rows = []
    for unit, label in (("chars", "characters"), ("tokens", "whitespace tokens")):
        v = lengths[unit]
        rows.append(
            [
                label,
                num(v["mean_a"]),
                num(v["mean_b"]),
                num(v["median_a"]),
                num(v["median_b"]),
                num(v["ks_statistic"]),
                num(v["ks_p"]),
                num(v["cliffs_delta"], 3),
            ]
        )
    header = ["length in", "mean A", "mean B", "median A", "median B", "KS D", "KS p"]
    header.append("Cliff's delta")
    lines = ["## Lengths", ""] + table(header, rows)
    lines += [
        "",
        "KS D is the largest gap between the two length distributions (0 to 1). "
        "Cliff's delta is P(B longer) - P(A longer).",
    ]
    return lines


def render_labels(labels: dict) -> list[str]:
    rows = [
        [
            cell(category),
            str(labels["counts_a"][i]),
            f"{labels['shares_a'][i]:.3f}",
            str(labels["counts_b"][i]),
            f"{labels['shares_b'][i]:.3f}",
        ]
        for i, category in enumerate(labels["categories"])
    ]
    lines = ["## Label shares", ""]
    lines += table(["label", "count A", "share A", "count B", "share B"], rows)
    lines += [
        "",
        f"G = {num(labels['g_statistic'])} on {labels['dof']} degrees of freedom, "
        f"p = {num(labels['g_p'])}; total variation = {num(labels['total_variation'])}.",
    ]
    return lines


def render_features(features: dict) -> list[str]:
    log_odds = features["log_odds"]
    lines = [
        "## Features: weighted log-odds",
        "",
        f"Informative Dirichlet prior with total {log_odds['prior_total']:g}; features with "
        f"fewer than {log_odds['min_count']} occurrences are not listed. Positive z: more "
        "typical of B.",
    ]
    for key, title in (("more_in_b", "More typical of B"), ("more_in_a", "More typical of A")):
        rows = [
            [
                cell(r["feature"]),
                str(r["count_a"]),
                str(r["count_b"]),
                f"{r['log_odds']:.3f}",
                f"{r['z']:.2f}",
            ]
            for r in log_odds[key]
        ]
        lines += ["", f"{title}:", ""]
        lines += table(["feature", "count A", "count B", "log-odds", "z"], rows)
    weights = features["classifier_weights"]
    lines += [
        "",
        "## Features: classifier weights",
        "",
        "`w:` marks a word n-gram, `c:` a character n-gram inside a padded word. "
        "A bucket shared by several names shows its most frequent one.",
    ]
    for key, title in (("toward_b", "Pushing towards B"), ("toward_a", "Pushing towards A")):
        rows = [
            [cell(r["feature"]), f"{r['weight']:.3f}", str(r["names_in_bucket"])]
            for r in weights[key]
        ]
        lines += ["", f"{title}:", ""]
        lines += table(["feature", "weight", "names in bucket"], rows)
    records = features["records"]
    lines += [
        "",
        "## Records",
        "",
        "Held-out probability of side B from the cross-fitted classifier, by record id.",
        "",
    ]
    titles = (("typical_of_a", "Most typical of A"), ("typical_of_b", "Most typical of B"))
    for key, title in titles:
        listed = ", ".join(f"{cell(r['id'])} ({r['p_b']:.3f})" for r in records[key])
        lines.append(f"- {title}: {listed}")
    return lines


def render_report(report: dict) -> str:
    a, b = report["a"], report["b"]
    lines = [
        f"# shiftscan report: {a['name']} vs {b['name']}",
        "",
        f"A: {a['name']} ({a['records']} records). B: {b['name']} ({b['records']} records).",
        "",
        f"Verdict: {report['tests']['verdict']}",
        "",
    ]
    lines += render_tests(report["tests"]) + [""]
    lines += render_baseline(report["baseline"]) + [""]
    lines += render_lengths(report["lengths"]) + [""]
    if report.get("labels"):
        lines += render_labels(report["labels"]) + [""]
    lines += [
        "## Unigram Jensen-Shannon divergence",
        "",
        f"{num(report['unigram_jsd'])} bits over all records of both datasets. It grows as "
        "samples shrink, so compare the matched-size row in the baseline table.",
        "",
    ]
    lines += render_features(report["features"])
    return "\n".join(lines)
