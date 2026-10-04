# Repository archaeology report

Audit date: 2026-10-04 (Asia/Shanghai). Audited snapshot: `main` at
`1c619604e061c1be0df9a6c722338ae29b152066`. This report describes that snapshot,
before continuation code is added. It follows `CODEX_ARCHAEOLOGY_PROMPT.md`.

## 1. Executive summary

This is a self-initiated NLP portfolio project intended to classify arXiv
computer-science abstracts into eight configured categories and eventually
compare classical TF-IDF baselines with a fine-tuned Transformer. It is not a
course assignment, competition submission, or published research project.

Development reached a usable, tested data-engineering layer: repository-aware
configuration, streaming JSONL reading, first-category selection, whitespace
cleaning, seeded per-class reservoir sampling, seeded splits, parquet writing,
and dataset-loading helpers. All **21 existing tests passed** in a newly created
environment. Synthetic data also exercised the whole existing data pipeline.

No baseline, Transformer, training script, shared evaluator, error analysis, or
substantive notebook exists at the audited snapshot. There are no verifiable
model-performance results, trained checkpoints, or real processed data in this
clone. The configured sampling cap is a target, not an observed dataset size.

The data layer has concrete limitations despite its passing tests: edited paths
can escape the repository; invalid UTF-8 aborts parsing; test fractions are not
validated; class shortfalls are silent; and duplicate IDs are not prevented from
crossing splits. Three of these limitations were already raised in PR reviews
and remain present in the audited code.

**History conclusion:** application development stopped immediately after PR #4
added the parquet dataset loader on 2026-06-09. The two subsequent commits on
2026-10-04 added the archaeology prompt and project-status documentation only.

## 2. Current architecture

The complete tracked tree at the audited snapshot is below. Git's internal
metadata is omitted. The clean audit environment and one-off probes were created
outside the repository, in a sibling `.audit-envs/` directory.

```text
arxiv-abstract-classification/
|-- .gitignore
|-- .gitkeep
|-- CODEX_ARCHAEOLOGY_PROMPT.md
|-- LICENSE
|-- PROJECT_STATUS.md
|-- README.md
|-- conftest.py
|-- requirements.txt
|-- config/
|   |-- .gitkeep
|   `-- config.yaml
|-- data/
|   |-- README.md
|   |-- raw/.gitkeep
|   `-- processed/.gitkeep
|-- notebooks/
|   `-- .gitkeep
|-- results/
|   |-- figures/.gitkeep
|   `-- metrics/.gitkeep
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- preprocess.py
|   |   `-- dataset.py
|   `-- models/
|       `-- __init__.py
`-- tests/
    |-- .gitkeep
    |-- test_config.py
    |-- test_preprocess.py
    `-- test_dataset.py
```

Real data flow:

```text
config/config.yaml
    -> load_config(): repository-relative paths resolved to absolute Path objects
    -> manually supplied data/raw/arxiv-metadata-oai-snapshot.json
    -> iter_records(): one JSON object per line
    -> primary_category(): first categories token
    -> filter configured classes, nonempty string IDs, nonempty abstracts
    -> clean_text(): collapse whitespace
    -> collect_samples(): seeded per-class reservoirs, capped at 7,000 each
    -> stratified_split(): per-class seeded shuffle, train/val floors, test remainder
    -> write_splits(): train.parquet, val.parquet, test.parquet
    -> load_split()/get_texts_labels()/get_texts_label_ids()
    -> no downstream model or evaluation implementation yet
```

The configured order is `cs.AI`, `cs.LG`, `cs.CL`, `cs.CV`, `cs.CR`, `cs.RO`,
`cs.DS`, `cs.SE`; integer labels follow that order. Defaults request seed `42`
and split fractions `0.8/0.1/0.1`. `max_seq_length: 256`,
`transformer_name: distilbert-base-uncased`, and `baseline: tfidf_logreg` are
currently unused model settings.

There is no installed-package definition, CLI entry point, `scripts/` directory,
or `if __name__ == "__main__"` runner. Preprocessing currently requires calling
`run_preprocessing()` from Python. All package `__init__.py` files are empty.

The label rule actually implemented is **the first token in `categories`**.
This audit did not independently establish that ordering's meaning in a real
Kaggle snapshot; document the operational rule and verify it against the chosen
source before making claims about authoritative primary-category labels.

## 3. Implemented vs planned matrix

Statuses refer to the audited `main`, not future work: COMPLETE means the stated
capability is implemented and verified; PARTIAL means material gaps remain;
PLANNED ONLY means documentation/configuration names it without implementation;
ABSENT means no actual implementation or meaningful result specification exists.

| Component | Status | Evidence | Tests | Missing work |
|---|---|---|---|---|
| Repository scaffold and project metadata | COMPLETE | MIT `LICENSE`, README, requirements, directory placeholders; PR #1 | File-tree/license/ignore inspection; no scaffold unit tests | No missing scaffold work; portfolio documentation remains unfinished |
| Path-safe configuration | PARTIAL | `src/config.py:7-23` anchors defaults to the repository | 4 configuration tests pass; external-CWD probe passes | Reject absolute/escaping configured paths and invalid settings; custom config-file paths themselves are CWD-relative |
| Raw arXiv metadata preprocessing | PARTIAL | `src/data/preprocess.py:12-44,125-143` streams JSONL and filters/cleans records | 9 preprocessing tests pass; synthetic pipeline succeeds | Invalid UTF-8 handling, processing statistics/source provenance, real-data verification |
| Balanced per-class sampling | PARTIAL | `collect_samples()` uses a seeded capped reservoir for each class | Existing cap/repeatability tests; 20,000-record one-pass probe | A cap does not ensure equal counts when classes are underfilled; no shortfall report or duplicate policy |
| Train/validation/test splitting | PARTIAL | `stratified_split()` shuffles within class; floors train/val sizes | Existing deterministic/disjoint-unique-ID test; repeated eight-class pipeline | Validate fractions, handle tiny classes explicitly, detect ID/text leakage |
| Parquet dataset loading | COMPLETE | `src/data/dataset.py` implements split/schema checks, label validation, aligned text/label lists | 8 dataset tests pass | Type/null/duplicate validation and cross-split checks would harden the interface; current advertised helpers work |
| TF-IDF + Logistic Regression baseline | PLANNED ONLY | `models.baseline: tfidf_logreg`; `PROJECT_STATUS.md` | None | Vectorizer/classifier pipeline, fit/predict interface and tests |
| TF-IDF + Linear SVM baseline | PLANNED ONLY | Explicitly listed as unfinished in `PROJECT_STATUS.md` | None | LinearSVC pipeline and tests |
| Baseline training scripts | PLANNED ONLY | Next-milestone list in `PROJECT_STATUS.md`; no runner files | None | Train-only fitting, validation selection, controlled held-out testing, saved run artifacts |
| Shared evaluation utilities | PLANNED ONLY | Next-milestone list; no evaluation module | None | Explicit class-order metrics and common artifact contract |
| Confusion matrix generation | PLANNED ONLY | Unfinished capability in `PROJECT_STATUS.md`; figures directory is empty | None | Ordered numeric matrices; later readable plots |
| Prediction artifact saving | PLANNED ONLY | Next-milestone list in `PROJECT_STATUS.md` | None | ID/text/true/predicted-label artifacts and alignment checks |
| Error analysis | PLANNED ONLY | README future-work section and project-status goals | None | Inspectable saved errors, category-level comparison, evidence-backed explanations |
| DistilBERT dataset/tokenization layer | PLANNED ONLY | DistilBERT name and maximum length in config; project-status list | None | Tokenization adapter and model-ready batches; offline tests |
| DistilBERT fine-tuning | PLANNED ONLY | Config model name; explicit project goal | None | Transformer dependencies, trainer, checkpoints and measured runs |
| Optional RoBERTa/SciBERT experiments | PLANNED ONLY | Named as unfinished in `PROJECT_STATUS.md` and the archaeology prompt | None | Defined experimental scope and implementations after the core comparison |
| Interpretability | PLANNED ONLY | README explicitly calls this future work | None | Scoped methods and reproducible analysis after trained models exist |
| Notebooks | PLANNED ONLY | `notebooks/.gitkeep` only; status document names substantive notebooks as unfinished | None | Optional exploratory/comparison notebooks; core code should remain usable without them |
| CI / GitHub Actions | PLANNED ONLY | No `.github/` files; GitHub API reports 0 workflows; status document lists CI as unfinished | None | Automated clean-install/root test job, ideally across supported platforms |
| Reproducibility / one-command workflow | PARTIAL | Seeded data functions, root-aware defaults, requirements, synthetic tests | 21 tests pass from root and external CWD | Version/environment specification, runnable workflow, manifests and usable README commands |
| Actual experimental results | ABSENT | No trained model, populated results directory, tags, or release assets observed | No model experiment to verify | Obtain real source data, train/evaluate models, preserve provenance and actual measurements |
| README claims versus reproducible behavior | PARTIAL | README uses future tense and explicitly reports no results | Source/document comparison | Document existing label/split rules and runnable data pipeline; distinguish planned title capabilities from current ones |

## 4. Test and reproducibility audit

### Environment and recorded verification

The clone was initially clean, on `main`, tracking `origin/main`. Both fetch and
push remotes were `https://github.com/ZhangHanai/arxiv-abstract-classification.git`.
`git ls-remote --symref origin HEAD` and GitHub's repository API both identified
`main` as the default branch. Git version was `2.55.0.windows.3`.

A fresh Windows virtual environment was created with **Python 3.13.5**, without
system site-packages, outside the repository. The command examples below use
portable checkout-relative paths in place of the original machine-local paths;
the recorded results are unchanged:

```powershell
python -m venv ../.audit-envs/arxiv-abstract-classification-20261004
$auditPython = '../.audit-envs/arxiv-abstract-classification-20261004/Scripts/python.exe'
& $auditPython -m pip install -r requirements.txt
$env:PYTHONDONTWRITEBYTECODE = '1'
& $auditPython -m pytest -q -p no:cacheprovider
& $auditPython -m pip check
& $auditPython -m pip freeze
```

Observed root result: **21 passed in 9.74s**; `pip check` reported
`No broken requirements found.` The existing tests are exactly:

| Existing test file | Tests observed passing |
|---|---:|
| `tests/test_config.py` | 4 |
| `tests/test_preprocess.py` | 9 |
| `tests/test_dataset.py` | 8 |
| Total | 21 |

The same suite was invoked from another directory outside the repository. An
equivalent invocation, with variables captured before changing directories, is:

```powershell
$repoPath = (Get-Location).Path
$auditPython = (Resolve-Path $auditPython).Path
# Change to an external directory, then run:
& $auditPython -B -m pytest "$repoPath/tests" --rootdir=$repoPath -q -p no:cacheprovider
```

Observed external-CWD result: **21 passed in 1.92s**. Root `conftest.py` inserts
the repository into `sys.path`. Separately, changing CWD before `load_config()`
returned the same default configuration and resolved paths. This verifies the
default configuration and test import behavior on this Windows installation;
it is not evidence of testing Linux/macOS or arbitrary execution entry points.

Installed application/test requirements were PyYAML `6.0.3`, pandas `3.0.6`,
pyarrow `25.0.1`, scikit-learn `1.9.1`, and pytest `9.1.1`. The complete observed
dependency resolution from `pip freeze` was:

```text
cloudpickle==3.1.2
colorama==0.4.6
iniconfig==2.3.0
joblib==1.6.0
narwhals==2.26.0
numpy==2.5.3
packaging==26.3
pandas==3.0.6
pluggy==1.6.0
pyarrow==25.0.1
Pygments==2.21.0
pytest==9.1.1
python-dateutil==2.9.0.post0
PyYAML==6.0.3
scikit-learn==1.9.1
scipy==1.18.1
six==1.17.0
threadpoolctl==3.7.0
tzdata==2026.5
```

`requirements.txt` is sufficient for the code currently present and its tests
in this clean environment. It is unpinned and does not specify a Python version,
so it does not recreate this exact environment on its own. scikit-learn is a
declared but unused future dependency at the audited snapshot. Transformer
dependencies are not present and are not needed for the current data layer.

### Additional forensic probes actually run

One-off probes were saved outside the repository as
`../.audit-envs/arxiv-abstract-classification-20261004/audit_checks.py`
and executed with the clean interpreter. They are separate from the 21 existing
tests; no new regression tests or application edits preceded this report.

| Probe | Observed outcome | Interpretation |
|---|---|---|
| Custom YAML paths using `../audit-escape` and an absolute temporary directory | Both accepted; both resolved outside `PROJECT_ROOT` | Default path portability works, but containment is not enforced (`src/config.py:18-21`) |
| JSONL containing an invalid UTF-8 byte between JSON records | `UnicodeDecodeError`; iteration aborted | Decode failure occurs before the parser's `try` block (`preprocess.py:16-24`) |
| 10 samples, fractions `train=0.8`, `val=0.1`, `test=0.6` | 8 train, 1 val, 1 test; no error | Test gets the remainder and configured proportions are not validated |
| Same split call with the `test` key entirely omitted | Still 8/1/1; no error | `splits['test']` is never read |
| 10 records sharing one paper ID | The same ID appeared in train, validation, and test | Splitting is by row, without ID deduplication or group protection |
| Classes with 10, 2, and 0 eligible records; cap 10 | Reservoir counts 10, 2, 0; validation contained no examples of the latter two classes | The implementation caps classes but does not guarantee balance or full class coverage |
| One-pass generator of 20,000 synthetic records, 2 classes, cap 32 | Consumed once; retained 32 per class | Supports the streaming/capped-reservoir design; not a large real-file benchmark |
| Sampling and splitting around a saved global RNG state | Global `random` state unchanged | Both functions use independent `random.Random(seed)` instances |
| Synthetic JSONL: 160 unique records, all 8 configured classes, cap 10 | 64 train, 8 validation, 8 test; all IDs disjoint; label IDs 0-7 | Whole existing pipeline and loader work for well-formed synthetic input |
| Repeat of that eight-class pipeline with identical input/config | Identical DataFrame contents and parquet bytes in this environment | Deterministic local repeat; not a cross-version byte-identity guarantee |

Memory behavior is also visible in the code: `iter_records()` reads lines, not
the whole file; `collect_samples()` retains at most `number_of_classes * cap`
sample rows for valid positive caps. Splitting and DataFrame creation temporarily
copy the selected sample, not the entire raw corpus. Memory still depends on
line/abstract lengths. No peak-memory or real-corpus throughput measurement was
made.

Randomness is deterministic for the same seed, record order, class order, and
environment. Changing input order or selected class ordering can change outcomes
because each stage shares one local generator across classes. There are no model
seeds to inspect because no model code exists.

Unique-ID synthetic inputs split without overlap. Duplicated IDs can leak as
demonstrated; identical cleaned abstracts with different IDs are also not
checked anywhere in the code. The audit did not inspect a real snapshot for
either kind of duplication. No vectorizer exists, so train/test vocabulary
leakage cannot yet be assessed; future training must fit preprocessing only on
training texts.

### Data, generated artifacts, and documentation

The local `data/raw`, `data/processed`, `results/figures`, `results/metrics`, and
`notebooks` directories contained only their placeholders. No huge dataset was
downloaded. The audit therefore verifies code and synthetic behavior, **not a
real arXiv end-to-end run or model performance**.

`git check-ignore` confirmed that raw/processed data and checkpoint paths are
ignored while `.gitkeep` placeholders are retained. These intentional exclusions
explain why data or checkpoints need not be in Git; they do not prove that any
were ever generated. `results/metrics/*.json` and `results/figures/*.png` are
not ignored, yet no such tracked or local results were found. GitHub reported
zero Actions workflows; no release assets or tags were found. No evidence of
training was found in any of the available commits or PRs.

README language is mostly honest: models, error analysis, and interpretability
are described as planned/future, and it explicitly says no results are reported.
Its broad title is aspirational. Its claim that target labels/splits will be
specified later is stale: eight classes, first-token labeling, and 80/10/10
defaults already exist. It omits working preprocessing/test commands and the
fixed raw filename now required by the configuration. `PROJECT_STATUS.md`
accurately identifies the stopping point, but its broad "balanced"/"path-safe"
claims require the qualifications demonstrated above.

## 5. Historical reconstruction

The full available history contains 11 commits: initialization, four feature
commits, four merges, and two later documentation commits. All fetched feature
branches are represented in that history; no additional application implementation
was found. All four PRs are closed and merged into `main`; no other PR or
non-PR issue was returned by the repository API.

Dates/times below use Asia/Shanghai. Historical test claims are attributed to PR
bodies; the current 21-test results were independently observed in section 4.

| Date/time | Commit(s) | Development event |
|---|---|---|
| 2026-06-09 16:50:43 | `42a9239` | Initialized the repository with a root `.gitkeep` |
| 2026-06-09 17:16:23 / 17:16:42 | `b15325a` / `96639a9` | Scaffold feature commit / PR #1 merge |
| 2026-06-09 17:23:50 / 17:24:19 | `fa5fb57` / `0421f92` | Configuration feature commit / PR #2 merge |
| 2026-06-09 17:35:30 / 17:36:50 | `5cac055` / `9ee86db` | Preprocessing feature commit / PR #3 merge |
| 2026-06-09 17:45:40 / 17:46:49 | `cf58723` / `4624865` | Dataset-loader feature commit / PR #4 merge |
| 2026-10-04 15:52:58 | `ee5fcdb` | Added `CODEX_ARCHAEOLOGY_PROMPT.md` only |
| 2026-10-04 15:53:29 | `1c61960` | Added `PROJECT_STATUS.md` only |

### PR #1: scaffold repository structure and project metadata

[PR #1](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/1)
created `.gitignore`, `LICENSE`, `README.md`, `requirements.txt`,
`data/README.md`, directory placeholders under config/data/notebooks/results/tests,
and the empty `src`, `src/data`, and `src/models` package files (15 files).

- Purpose: establish metadata and a clean portfolio layout without application logic.
- Tests added: none. The PR reported scaffold, whitespace, and ignore-rule checks.
- Capability made available: documented manual dataset acquisition and basic layout.
- Next unfinished step: configuration and executable data processing.

### PR #2: repository-aware configuration

[PR #2](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/2)
added `config/config.yaml`, `src/config.py`, `conftest.py`, and
`tests/test_config.py`, and added pytest to `requirements.txt` (5 files).

- Purpose: load YAML defaults independently of CWD and resolve project paths.
- Tests added: 4, covering defaults, absolute resolved paths, default containment,
  and expected subdirectories. The PR recorded 4 passing tests.
- Capability made available: centralized seed/classes/data/model/path settings.
- Next unfinished step: streaming raw-data preprocessing.
- Review finding: an automated review after merge warned that absolute and `..`
  paths can escape the root. Current tests cover only default paths; the finding
  remains reproducible in the audited snapshot.

### PR #3: streaming preprocessing

[PR #3](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/3)
added `src/data/preprocess.py` and `tests/test_preprocess.py`, and added
`data.raw_filename` to `config/config.yaml` (3 files).

- Purpose: turn a line-oriented arXiv snapshot into capped class samples and
  deterministic train/validation/test parquet splits.
- Tests added: 9, covering JSON parsing, first-token categories, whitespace,
  invalid/non-target filtering, cap/repeatability, splits, parquet writing, and
  a synthetic full preprocessing run. The PR recorded 9 new and 13 total passes.
- Capability made available: a callable streaming-to-parquet pipeline.
- Next unfinished step: stable split-loading interfaces for model code.
- Review findings: validate/honor test proportions and catch decoding errors at
  the stream boundary. Both were raised just after merge and remain present.

### PR #4: dataset-loading layer

[PR #4](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/4)
added `src/data/dataset.py` and `tests/test_dataset.py` (2 files).

- Purpose: read processed parquet splits and expose ordered string/integer labels.
- Tests added: 8, covering mapping order, parquet loading, invalid split names,
  missing columns, row alignment, integer encoding, and accepted/rejected labels.
  The PR recorded 8 dataset passes and a passing full suite; the current full
  suite was independently observed to contain 21 passes.
- Capability made available: the data-consumption interface expected by baselines
  and later Transformer training.
- Next unfinished step: shared evaluation and train-only classical baselines.
- No inline review comments were returned for this PR.

PR discussions contained no additional issue comments. The reviewed code matches
the current implementation; later commits did not address the three review
findings or add downstream ML capabilities.

## 6. Gaps and technical debt

These are observed issues or directly missing deliverables, not speculative
claims of failures on unseen real data.

1. **Path containment is not enforced** (`src/config.py:18-21`): custom absolute
   or traversal values are accepted outside the project. Preserve the default
   root-aware approach while validating configured paths and config structure.
2. **Malformed encoding aborts the stream** (`preprocess.py:16-24`): the decoding
   exception occurs outside the protected JSON parsing block. Add a documented
   per-line decode policy and a regression case.
3. **Invalid fractions are silently accepted** (`preprocess.py:95-101`): `test`
   is unused and fractions/ranges are not validated. Document rounding and
   reject inconsistent proportions, negative values, and unusable sample counts.
4. **"Balanced" means capped, conditionally balanced** (`collect_samples()`):
   missing/underrepresented classes silently remain short and can disappear
   from validation. Report counts and choose a documented shortfall policy.
5. **No duplicate or overlap policy exists**: duplicated paper IDs demonstrably
   cross splits; identical cleaned text is unchecked. Add ID/group protection
   and observable cross-split checks before publishing model comparisons.
6. **Dataset validation is narrow** (`dataset.py:20-60`): required columns and
   label membership are checked, but empty splits, null/non-string text/IDs,
   repeated class names, and duplicate rows are not validated. These checks are
   needed before training; ordinary well-formed loading already works.
7. **No executable, documented experiment workflow or provenance**: there are
   only callable functions, unpinned requirements, no source fingerprint/count
   manifest, and no CI. Synthetic tests cannot establish a real source's label
   meaning, availability, balance, or performance.
8. **The complete model/evaluation pipeline is still missing**: no learner,
   trainer, evaluator, predictions, analysis, or measured benchmark exists.
   Model-related config values alone do not implement those capabilities.
9. **README task/reproducibility details are stale or absent**: it describes label
   and split choices as undecided even though they are encoded, and provides no
   runnable commands. Avoid widening claims before the pipeline exists.

The absence of large data/checkpoints from Git is appropriate, not technical
debt. No working code needs a broad rewrite to address these gaps.

## 7. Resume honesty check

Safe, specific claims supported by this audit:

- "Implemented streaming JSONL preprocessing and deterministic per-class
  reservoir sampling for an arXiv abstract-classification portfolio project."
- "Built repository-aware YAML configuration, seeded dataset splitting, parquet
  exports, and aligned string/integer label-loading utilities."
- "Verified 21 automated data/configuration tests in a clean Python environment,
  including execution from the repository root and an external directory."
- "Validated the data pipeline using small synthetic datasets."

Qualify any claim of balanced or leakage-free data: current sampling is balanced
only when class reservoirs fill, and duplicate-ID leakage is possible. Do not
claim a large real snapshot was processed merely because the reader is streaming.
Default path portability is verified; general path containment is not.

Premature or unsupported claims at the audited snapshot:

- trained TF-IDF/Logistic Regression, Linear SVM, or any Transformer;
- fine-tuned DistilBERT, RoBERTa, or SciBERT;
- achieved any accuracy/F1, training time, or benchmark improvement;
- processed a particular real dataset size or established real-data balance;
- completed a baseline-versus-Transformer comparison, error analysis,
  interpretability study, demo, or portfolio-ready end-to-end project;
- published research, competed in a modeling contest, or completed this as a
  course assignment.

No private prior work is inferred either way: the available repository/history
simply provides no verifiable evidence of those achievements.

## 8. Recommended continuation plan

The core goal remains raw data -> reproducible preprocessing -> classical
baselines -> shared evaluation -> DistilBERT -> measured comparison -> error
analysis -> reproducible results. Build the evaluator before training so both
baseline implementations use the same label ordering and metric definitions.
Eight small, ordered PRs are proposed; names are proposals, not existing PRs.

### Proposed PR 1: shared classification evaluation and traceable artifacts

- Objective: establish the first missing core interface before model training.
- Files: add `src/evaluation.py` and `tests/test_evaluation.py`; document the
  evaluation contract in this report or a concise usage document.
- Required tests: hand-calculated accuracy/macro-F1/weighted-F1, explicit class
  order, absent classes, invalid/unaligned labels, confusion-matrix layout,
  round-trip prediction artifacts, preservation of row order and source data.
- Acceptance: one model-independent API produces JSON metrics, a labeled numeric
  confusion matrix, and ID/text/true/predicted-label artifacts; no model fitting,
  data downloads, or invented experiment results.
- Dependencies: existing PR #4 only. Plot rendering can follow with training.

### Proposed PR 2: close existing data-safety gaps and expose preprocessing

- Objective: make inputs/splits auditable before baseline runs.
- Files: update `src/config.py`, `src/data/preprocess.py`, `src/data/dataset.py`,
  their tests, `data/README.md`; add a small preprocessing CLI and manifest.
- Required tests: escaping paths, invalid fractions, malformed UTF-8, class
  shortfalls, duplicate IDs/text overlap, tiny/empty/null data, deterministic
  manifest/count output, CLI execution with a temporary snapshot.
- Acceptance: fail clearly on unsupported config/data, handle malformed lines
  under a documented policy, enforce/document duplicate and shortfall choices,
  report actual per-class/split counts and source fingerprint, and provide a
  tested root command. Verify source category ordering before interpreting it.
- Dependencies: existing data layer; PR 1's class-order contract should be shared.

### Proposed PR 3: TF-IDF linear baselines and a training runner

- Objective: complete train-only Logistic Regression and Linear SVM baselines.
- Files: add `src/models/baselines.py`, a baseline training module/CLI, matching
  tests; extend config with explicit model/vectorizer parameters and artifact
  ignore rules where necessary.
- Required tests: both pipelines fit/predict on tiny local data, validation/test
  vocabulary never enters the fitted vectorizer, seed repeatability, stable
  class mapping, rejected split overlap, and evaluator/artifact integration.
- Acceptance: a root command trains each baseline only on train, uses validation
  for choices, reports held-out test results after choices are fixed, and saves
  parameters/seed/versions/data fingerprint alongside real computed outputs.
- Dependencies: PRs 1-2. No Transformer before this pipeline works.

### Proposed PR 4: reproducible baseline workflow and CI

- Objective: make the minimum complete classical pipeline reproducible.
- Files: add a small workflow runner and `.github/workflows/tests.yml`; update
  README, dependency/environment specification, and integration tests.
- Required tests: clean installation and complete raw-to-baseline artifact run
  on synthetic fixtures; root execution; supported-platform CI checks.
- Acceptance: one documented command reaches saved evaluation from a manually
  provided raw snapshot, CI is green, and fixture measurements are explicitly
  distinguished from real benchmark results. Keep generated large artifacts local.
- Dependencies: PRs 1-3. Real runs require separately obtained source data.

### Proposed PR 5: DistilBERT tokenization/data adapter

- Objective: adapt the existing split/label interface for DistilBERT inputs.
- Files: add a Transformer data module and its tests; add separately documented
  optional Transformer dependencies and config settings.
- Required tests: stub-tokenizer truncation/padding, attention masks, configured
  label IDs, text/label alignment, split isolation, offline behavior.
- Acceptance: model-ready batches honor `max_seq_length` and share the same
  task/splits; ordinary tests need neither a download nor GPU. This is not training.
- Dependencies: PR 4's working baseline/data contract.

### Proposed PR 6: DistilBERT fine-tuning runner

- Objective: implement controlled fine-tuning using the shared evaluator.
- Files: add Transformer model/training modules and runner tests; extend config
  and optional dependency/environment documentation.
- Required tests: lightweight/stub trainer integration, deterministic seed
  propagation, validation-only model selection, checkpoint/result paths,
  prediction alignment; explicit optional real-model smoke procedure.
- Acceptance: a documented training command with traceable run metadata and
  held-out evaluation; only claim an actual fine-tuning run after executing and
  preserving evidence of it. Routine tests remain offline.
- Dependencies: PRs 1-5; actual runs need data and adequate compute.

### Proposed PR 7: comparison and error-analysis artifacts

- Objective: compare the same task/splits and inspect representative errors.
- Files: add comparison/error-analysis modules and tests; populate only genuinely
  computed small result tables/figures after runs exist.
- Required tests: matching dataset/split fingerprints, metric schema/label-order
  compatibility, error filtering, ID-aligned model joins, absent-class handling.
- Acceptance: baseline/Transformer comparison uses identical held-out examples
  and recorded runs; examples and explanations are grounded in saved predictions.
- Dependencies: PRs 3 and 6 with verifiable model runs for a real comparison.

### Proposed PR 8: measured portfolio results and honest documentation

- Objective: turn the runnable pipeline into an evidence-backed portfolio.
- Files: update README, `PROJECT_STATUS.md`, `data/README.md`, result summaries,
  limitations, and an optional compact reproduction notebook.
- Required verification: rerun documented commands and trace every published
  score/table/example to preserved run metadata and prediction artifacts.
- Acceptance: clearly separate completed capabilities, actual measurements,
  limitations, and remaining plans; explain first-token labels, sampling bias,
  duplicates, source/version, and hardware for reported timing measurements.
- Dependencies: PR 7 and real runs. If data/compute are unavailable, document
  that limitation instead of replacing measurements with fabricated numbers.

RoBERTa/SciBERT, interpretability, demos, and extensive notebooks are stretch
goals after this core sequence. They are not dependencies for a complete first
comparison.

### Audit handoff

This report was finished and saved before changing any application code. The user
authorized continuing with the first missing core component after the report.
The first continuation increment is proposed PR 1: shared evaluation and traceable
artifacts. The snapshot classifications above remain a record of the original
`main`; continuation verification will be recorded separately below.

### Continuation result: shared evaluation completed in the working branch

After saving the audit, the first proposed increment was implemented on the
local branch `codex/archaeology-and-evaluation`. At the original handoff, no
commits or remote writes had been made. The GitHub default branch was `main`,
and the origin URL was unchanged.
No pre-existing tracked project file was modified.

Exact repository additions:

- `ARCHAEOLOGY_REPORT.md`: the original audit plus this continuation record.
- `src/evaluation.py`: `evaluate_predictions(y_true, y_pred, classes)` and
  `save_evaluation(dataframe, predictions, classes, output_dir)`.
- `tests/test_evaluation.py`: 42 test cases for the new contract and its
  integration with the existing data pipeline.

The evaluator returns JSON-ready accuracy, macro F1, support-weighted F1,
per-class precision/recall/F1/support, and a numeric confusion matrix. Class
ordering is explicit; macro F1 includes every configured class, including absent
classes, and undefined class metrics are zero. Matrix rows represent true labels
and columns predicted labels.

The saver consumes the existing loader's `id/text/label` DataFrame schema. It
preserves input row order and text, rejects unaligned/unknown labels and invalid
or repeated IDs, and does not mutate the source DataFrame. It writes:

- `metrics.json`;
- `confusion_matrix.csv`, with labeled axes;
- `predictions.parquet`, with `id`, `text`, `true_label`, `predicted_label`, and
  `correct` columns.

The caller must use a separate destination for each run/split. To connect a
future model, pass its ordered predictions alongside `load_split('val', config)`
and an output directory below a resolved configuration path. This evaluator
does not fit models, select hyperparameters, render plots, or validate overlap
with other splits. Source/run provenance belongs in the subsequent workflow.

Tests were run in the same clean environment from the repository root, with
bytecode and pytest's cache provider disabled:

| Recorded command (environment executable) | Observed result |
|---|---|
| `pytest tests/test_evaluation.py -q -p no:cacheprovider` before the last two test additions | 40 passed in 14.18s |
| `pytest tests/test_evaluation.py -q -p no:cacheprovider` on the final code | 42 passed in 4.48s |
| `pytest -q -p no:cacheprovider` on the final code | 63 passed in 5.14s |

Here `pytest` was the audit environment's `Scripts/pytest.exe` executable.
The 63 final passes are the original 21 plus 42 evaluation cases. Hand-calculated
fixtures verify scores and matrix orientation; parquet/CSV/JSON round trips
verify alignment and preservation. An integration test exercised synthetic
JSONL -> existing preprocessing -> parquet loader -> evaluation artifacts.
All generated data/results in these checks were temporary test fixtures.

**Now genuinely complete in the working branch:** shared classification metrics,
labeled numeric confusion matrices, and traceable prediction-artifact saving,
verified with synthetic inputs. No baseline or Transformer was trained, and no
real experimental result was produced. Matrix plotting, training, source/run
provenance, and the original data-safety issues remain incomplete.

**Next recommended PR:** proposed PR 2, closing the verified configuration,
encoding, split-fraction, shortfall, and duplicate/overlap gaps and adding a
documented preprocessing command. Then implement the classical baseline runner
in proposed PR 3. The user authorized the first continuation increment for this
run; the remaining proposed PRs were recommendations for subsequent work.

### Preservation review: 2026-10-04

The three additions were reviewed again before committing. Machine-local
absolute paths in this report were replaced with portable examples. No model
results, datasets, checkpoints, generated evaluation outputs, or unrelated
changes are included. The audit's one-off probes were rerun and reproduced the
path, decode, fraction, duplicate-ID, shortfall, streaming, RNG, and synthetic
pipeline observations above. Installed versions match the recorded freeze;
`python -m pip check` reports `No broken requirements found.`

The complete suite was rerun with the audit interpreter and bytecode/cache
generation disabled: `python -B -m pytest -q -p no:cacheprovider` produced
**63 passed in 5.83s** (21 existing cases plus 42 evaluation cases). This result
is synthetic verification, not a real-data or trained-model benchmark.
