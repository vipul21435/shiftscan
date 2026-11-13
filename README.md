# shiftscan

Test whether two text datasets (training and evaluation, an old and a new collection, two
sources) come from the same distribution.

This is an early version. The full set of statistics is listed under "Not there yet".

## What works so far

- Hashed word n-gram (1 to 2) and character n-gram (3 to 5) features into a fixed number of
  buckets (CRC-32, so the same on every machine).
- An L2-regularised logistic regression written with numpy and scipy (L-BFGS on sparse
  matrices), checked in the tests against scikit-learn.
- A classifier two-sample test: equal-size subsampling with a seed, cross-fitted held-out
  accuracy and AUC, a binomial p-value and a permutation p-value.
- `shiftscan test A B --text FIELD` for JSON lines, CSV, TSV and (with the `parquet` extra)
  Parquet files, with Markdown or JSON output and a verdict line whose thresholds are flags.
- `--self-split` compares two random halves of one file, a control where no shift exists.

## Install

```
pip install git+https://github.com/vipul21435/shiftscan
```

## Try it

```
python examples/make_example_data.py
shiftscan test examples/data/billing.jsonl examples/data/login.jsonl --text text
```

Output of the second command on the synthetic example files (a planted topic shift):

```
# shiftscan test: billing.jsonl vs login.jsonl

- records: 200 and 200; 200 drawn from each side
- held-out accuracy: 0.995 (5 folds)
- held-out AUC: 1.000
- binomial p-value: 3.106e-116
- permutation p-value: 0.009901 (100 permutations)

Verdict: shift detected: p = 0.0099 < alpha = 0.05, held-out AUC 1.000 is at or above --min-auc 0.6
```

and of `shiftscan test examples/data/billing.jsonl --self-split --text text`:

```
- held-out accuracy: 0.475 (5 folds)
- held-out AUC: 0.491
- binomial p-value: 0.7816
- permutation p-value: 0.604 (100 permutations)

Verdict: no shift detected: p = 0.604 is not below alpha = 0.05 (held-out AUC 0.491)
```

With cross-fitting the folds share training data, so the binomial p-value is an
approximation; the permutation p-value repeats the whole procedure on shuffled side labels.

## Not there yet

- The kernel test (MMD), length and label comparisons, unigram Jensen-Shannon divergence.
- The split-half baseline for every statistic.
- Feature and record attribution (weighted log-odds, classifier weights, typical records).
- Results on real public data.

## Licence

MIT. Copyright (c) 2026 Vipul Raj Jha.

Built by Vipul Raj Jha · https://vipul21435.github.io
