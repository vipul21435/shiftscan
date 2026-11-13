"""Command line interface: ``shiftscan test | features | lengths | labels | report``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from . import report as render
from .features import FeatureConfig
from .io import Dataset, InputError, read_dataset
from .scan import (
    Options,
    baseline_section,
    describe,
    features_section,
    full_report,
    labels_section,
    lengths_section,
    split_halves,
    tests_section,
)

DEFAULTS = Options()


def _add_inputs(parser: argparse.ArgumentParser, text: bool = True) -> None:
    parser.add_argument("a", help="first file (JSON lines, CSV, TSV or Parquet)")
    parser.add_argument(
        "b", nargs="?", help="second file; leave out together with --self-split for a control"
    )
    parser.add_argument(
        "--text",
        action="append",
        required=text,
        metavar="FIELD",
        help="text field; repeat to join several fields",
    )
    parser.add_argument("--id", dest="id_field", metavar="FIELD", help="record id field")
    parser.add_argument(
        "--input-format",
        choices=["jsonl", "csv", "tsv", "parquet"],
        help="override the format guessed from the file suffix",
    )
    parser.add_argument(
        "--self-split",
        action="store_true",
        help="compare two random halves of the first file (a control with no shift)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for every random choice")
    parser.add_argument("--format", choices=["md", "json"], default="md", help="output format")
    parser.add_argument("--out", metavar="FILE", help="write the output here instead of stdout")


def _add_classifier(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULTS.max_records,
        help="records drawn from each side for the classifier (default %(default)s)",
    )
    parser.add_argument("--word-ngrams", type=int, default=2, help="longest word n-gram (0: off)")
    parser.add_argument(
        "--char-ngrams", type=int, default=5, help="longest character n-gram, from 3 (0: off)"
    )
    parser.add_argument("--buckets", type=int, default=2**18, help="number of hash buckets")
    parser.add_argument("--folds", type=int, default=DEFAULTS.folds, help="cross-fitting folds")
    parser.add_argument("--l2", type=float, default=DEFAULTS.l2, help="classifier L2 penalty")


def _add_tests(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULTS.permutations,
        help="label permutations for the classifier null (0: binomial p-value only)",
    )
    parser.add_argument(
        "--mmd-records",
        type=int,
        default=DEFAULTS.mmd_records,
        help="records per side for the kernel test (default %(default)s)",
    )
    parser.add_argument(
        "--mmd-permutations",
        type=int,
        default=DEFAULTS.mmd_permutations,
        help="permutations for the kernel test (default %(default)s)",
    )
    parser.add_argument("--kernel", choices=["rbf", "linear"], default=DEFAULTS.kernel)
    parser.add_argument("--alpha", type=float, default=DEFAULTS.alpha, help="significance level")
    parser.add_argument(
        "--min-auc",
        type=float,
        default=DEFAULTS.min_auc,
        help="held-out AUC from which a detected shift is called more than small",
    )


def _add_attribution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top", type=int, default=DEFAULTS.top, help="rows per list")
    parser.add_argument(
        "--prior-total",
        type=float,
        default=DEFAULTS.prior_total,
        help="total pseudo-count of the Dirichlet prior for the log-odds (default %(default)s)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULTS.min_count,
        help="occurrences a feature needs to be listed (default %(default)s)",
    )
    parser.add_argument(
        "--log-odds-ngrams", type=int, default=1, help="longest word n-gram scored by log-odds"
    )


def _add_baseline(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-repeats",
        type=int,
        default=DEFAULTS.baseline_repeats,
        help="random splits averaged for the split-half baseline (default %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shiftscan",
        description="Test whether two text datasets come from the same distribution.",
    )
    parser.add_argument("--version", action="version", version=f"shiftscan {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    test = sub.add_parser("test", help="classifier and kernel two-sample tests with a verdict")
    _add_inputs(test)
    _add_classifier(test)
    _add_tests(test)

    features = sub.add_parser("features", help="features and records that drive the difference")
    _add_inputs(features)
    _add_classifier(features)
    _add_attribution(features)

    lengths = sub.add_parser("lengths", help="length distributions against the split-half baseline")
    _add_inputs(lengths)
    _add_baseline(lengths)

    labels = sub.add_parser("labels", help="label shares: G-test and total variation")
    _add_inputs(labels, text=False)
    labels.add_argument("--label", required=True, metavar="FIELD", help="label field")
    _add_baseline(labels)

    report = sub.add_parser("report", help="everything, with the split-half baseline")
    _add_inputs(report)
    report.add_argument("--label", metavar="FIELD", help="label field (optional)")
    _add_classifier(report)
    _add_tests(report)
    _add_attribution(report)
    _add_baseline(report)
    return parser


OPTION_NAMES = (
    "max_records",
    "folds",
    "l2",
    "permutations",
    "mmd_records",
    "mmd_permutations",
    "kernel",
    "alpha",
    "min_auc",
    "top",
    "prior_total",
    "min_count",
    "log_odds_ngrams",
    "baseline_repeats",
)


def options_from(args: argparse.Namespace) -> Options:
    values: dict = {"seed": args.seed}
    if hasattr(args, "buckets"):
        word = (1, args.word_ngrams) if args.word_ngrams > 0 else (0, 0)
        char = (min(3, args.char_ngrams), args.char_ngrams) if args.char_ngrams > 0 else (0, 0)
        values["features"] = FeatureConfig(
            word_ngrams=word, char_ngrams=char, n_buckets=args.buckets
        )
    for name in OPTION_NAMES:
        if hasattr(args, name):
            values[name] = getattr(args, name)
    options = Options(**values)
    for name in ("max_records", "folds", "top", "baseline_repeats", "mmd_records"):
        if getattr(options, name) < 1:
            raise InputError(f"--{name.replace('_', '-')} must be at least 1")
    if not 0 < options.alpha < 1:
        raise InputError("--alpha must be between 0 and 1")
    return options


def load_pair(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    label_field = getattr(args, "label", None)
    text_fields = args.text or ([label_field] if label_field else [])
    first = read_dataset(args.a, text_fields, label_field, args.id_field, args.input_format)
    if args.self_split:
        if args.b:
            raise InputError("--self-split takes one file, two were given")
        return split_halves(first, np.random.default_rng((args.seed, 4)))
    if not args.b:
        raise InputError("give a second file, or --self-split to compare halves of the first")
    second = read_dataset(args.b, text_fields, label_field, args.id_field, args.input_format)
    if second.name == first.name:
        first.name, second.name = f"A:{first.name}", f"B:{second.name}"
    return first, second


def _emit(args: argparse.Namespace, payload: dict, lines: list[str]) -> None:
    if args.format == "json":
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    else:
        text = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text.rstrip("\n") + "\n")
    else:
        print(text.rstrip("\n"))


def _head(command: str, a: Dataset, b: Dataset) -> list[str]:
    return [
        f"# shiftscan {command}: {a.name} vs {b.name}",
        "",
        f"A: {a.name} ({len(a)} records). B: {b.name} ({len(b)} records).",
        "",
    ]


def cmd_test(args: argparse.Namespace) -> int:
    options = options_from(args)
    a, b = load_pair(args)
    section, _, _ = tests_section(a, b, options)
    payload = {"a": describe(a), "b": describe(b), "tests": section}
    lines = _head("test", a, b) + render.render_tests(section)
    lines += ["", f"Verdict: {section['verdict']}"]
    _emit(args, payload, lines)
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    options = options_from(args)
    options.mmd_permutations = 0
    a, b = load_pair(args)
    _, pooled, classifier = tests_section(a, b, options, permutations=0)
    section = features_section(a, b, pooled, classifier, options)
    payload = {"a": describe(a), "b": describe(b), "features": section}
    _emit(args, payload, _head("features", a, b) + render.render_features(section))
    return 0


def cmd_lengths(args: argparse.Namespace) -> int:
    options = options_from(args)
    a, b = load_pair(args)
    section = lengths_section(a, b)
    baseline = baseline_section(a, b, options, heavy=False)
    payload = {"a": describe(a), "b": describe(b), "lengths": section, "baseline": baseline}
    lines = _head("lengths", a, b) + render.render_lengths(section) + [""]
    lines += render.render_baseline(baseline, only=("length_ks_chars", "length_ks_tokens"))
    _emit(args, payload, lines)
    return 0


def cmd_labels(args: argparse.Namespace) -> int:
    options = options_from(args)
    a, b = load_pair(args)
    section = labels_section(a, b)
    baseline = baseline_section(a, b, options, heavy=False)
    payload = {"a": describe(a), "b": describe(b), "labels": section, "baseline": baseline}
    lines = _head("labels", a, b) + render.render_labels(section) + [""]
    lines += render.render_baseline(baseline, only=("label_total_variation",))
    _emit(args, payload, lines)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    options = options_from(args)
    a, b = load_pair(args)
    payload = full_report(a, b, options)
    _emit(args, payload, [render.render_report(payload)])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "test": cmd_test,
        "features": cmd_features,
        "lengths": cmd_lengths,
        "labels": cmd_labels,
        "report": cmd_report,
    }
    try:
        return handlers[args.command](args)
    except (InputError, ValueError) as exc:
        print(f"shiftscan: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
