"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .features import FeatureConfig
from .io import Dataset, InputError, read_dataset
from .scan import equal_size_subsample, pool, run_classifier_test, split_halves, verdict


def _add_inputs(parser: argparse.ArgumentParser, text: bool = True) -> None:
    parser.add_argument("a", help="first file (JSON lines, CSV, TSV or Parquet)")
    parser.add_argument(
        "b", nargs="?", help="second file; leave out together with --self-split for a control"
    )
    if text:
        parser.add_argument(
            "--text",
            action="append",
            required=True,
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


def _add_features(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-records",
        type=int,
        default=2000,
        help="records drawn from each side for the classifier test (default 2000)",
    )
    parser.add_argument("--word-ngrams", type=int, default=2, help="longest word n-gram (0: off)")
    parser.add_argument(
        "--char-ngrams", type=int, default=5, help="longest character n-gram, from 3 (0: off)"
    )
    parser.add_argument("--buckets", type=int, default=2**18, help="number of hash buckets")
    parser.add_argument("--folds", type=int, default=5, help="cross-fitting folds")
    parser.add_argument(
        "--permutations",
        type=int,
        default=100,
        help="label permutations for the classifier null (0: binomial p-value only)",
    )
    parser.add_argument("--l2", type=float, default=1.0, help="L2 penalty of the classifier")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level")
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.6,
        help="held-out AUC from which a detected shift is called more than small",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shiftscan",
        description="Test whether two text datasets come from the same distribution.",
    )
    parser.add_argument("--version", action="version", version=f"shiftscan {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("test", help="classifier two-sample test with a verdict line")
    _add_inputs(test)
    _add_features(test)
    return parser


def _feature_config(args: argparse.Namespace) -> FeatureConfig:
    word = (1, args.word_ngrams) if args.word_ngrams > 0 else (0, 0)
    char = (min(3, args.char_ngrams), args.char_ngrams) if args.char_ngrams > 0 else (0, 0)
    return FeatureConfig(word_ngrams=word, char_ngrams=char, n_buckets=args.buckets)


def load_pair(args: argparse.Namespace, label_field: str | None = None) -> tuple[Dataset, Dataset]:
    text_fields = getattr(args, "text", None) or []
    if not text_fields and label_field:
        text_fields = [label_field]
    first = read_dataset(args.a, text_fields, label_field, args.id_field, args.input_format)
    if args.self_split:
        if args.b:
            raise InputError("--self-split takes one file, two were given")
        return split_halves(first, np.random.default_rng(args.seed))
    if not args.b:
        raise InputError("give a second file, or --self-split to compare halves of the first")
    second = read_dataset(args.b, text_fields, label_field, args.id_field, args.input_format)
    if second.name == first.name:
        first.name, second.name = f"A:{first.name}", f"B:{second.name}"
    return first, second


def _emit(args: argparse.Namespace, payload: dict, markdown: str) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) if args.format == "json" else markdown
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text.rstrip("\n") + "\n")
    else:
        print(text.rstrip("\n"))


def cmd_test(args: argparse.Namespace) -> int:
    a, b = load_pair(args)
    rng = np.random.default_rng(args.seed)
    sample_a, sample_b = equal_size_subsample(a, b, args.max_records, rng)
    pooled = pool(sample_a, sample_b, _feature_config(args))
    result = run_classifier_test(
        pooled, folds=args.folds, permutations=args.permutations, seed=args.seed, l2=args.l2
    )
    p_value = result.permutation_p if result.permutation_p is not None else result.binomial_p
    line = verdict(p_value, result.auc, args.alpha, args.min_auc)
    payload = {
        "a": {"name": a.name, "records": len(a)},
        "b": {"name": b.name, "records": len(b)},
        "classifier": result.to_dict(),
        "verdict": line,
    }
    perm = "not run" if result.permutation_p is None else f"{result.permutation_p:.4g}"
    markdown = "\n".join(
        [
            f"# shiftscan test: {a.name} vs {b.name}",
            "",
            f"- records: {len(a)} and {len(b)}; {result.n_a} drawn from each side",
            f"- held-out accuracy: {result.accuracy:.3f} ({result.folds} folds)",
            f"- held-out AUC: {result.auc:.3f}",
            f"- binomial p-value: {result.binomial_p:.4g}",
            f"- permutation p-value: {perm} ({result.permutations} permutations)",
            "",
            f"Verdict: {line}",
        ]
    )
    _emit(args, payload, markdown)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"test": cmd_test}
    try:
        return handlers[args.command](args)
    except (InputError, ValueError) as exc:
        print(f"shiftscan: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
