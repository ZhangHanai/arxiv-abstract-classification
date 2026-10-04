# Codex Repository Archaeology Prompt

Use this prompt when resuming work on this repository with Codex.

---

You are taking over an old, partially completed portfolio project: `ZhangHanai/arxiv-abstract-classification`.

This is a self-initiated NLP portfolio project designed to strengthen a graduate-school application. It is NOT a course assignment, competition submission, or published research project. Do not imply that planned work was already completed, and do not fabricate experimental results.

Your first job is repository archaeology, not implementation.

## Phase 1 — Forensic audit

Inspect the entire repository before changing any application code.

Read:
- the complete current file tree;
- README.md;
- config/config.yaml;
- every file under src/;
- every file under tests/;
- requirements.txt and .gitignore;
- the full commit history and pull requests, especially PRs #1–#4.

Run the existing test suite from a clean environment if possible.

Reconstruct exactly what was actually implemented and what was merely planned.

For every planned component, classify it as one of:
- COMPLETE — implemented and tested in the current main branch;
- PARTIAL — some implementation exists but is incomplete;
- PLANNED ONLY — mentioned in docs/history but no implementation exists;
- ABSENT — neither implemented nor meaningfully specified.

At minimum audit these components:
1. repository scaffold and project metadata;
2. path-safe configuration;
3. raw arXiv metadata preprocessing;
4. balanced per-class sampling;
5. train/validation/test splitting;
6. parquet dataset loading;
7. TF-IDF + Logistic Regression baseline;
8. TF-IDF + Linear SVM baseline;
9. baseline training scripts;
10. shared evaluation utilities;
11. confusion matrix generation;
12. prediction artifact saving;
13. error analysis;
14. DistilBERT dataset/tokenization layer;
15. DistilBERT fine-tuning;
16. optional RoBERTa/SciBERT experiments;
17. interpretability;
18. notebooks;
19. CI / GitHub Actions;
20. reproducibility / one-command workflow;
21. actual experimental results;
22. README claims versus what the repository can currently reproduce.

## Verification rules

Do not trust old notes or README language without checking the code.

Specifically verify:
- which tests currently exist;
- the exact number of tests that pass now;
- whether tests run from the repository root;
- whether paths are portable;
- whether random seeds are actually used consistently;
- whether preprocessing can handle the large arXiv JSONL file without loading it all into memory;
- whether class balancing and stratified splits are deterministic;
- whether train/validation/test leakage is possible;
- whether requirements.txt is sufficient for the current code;
- whether any generated data, checkpoints, or metrics are intentionally absent because of .gitignore;
- whether README.md claims any functionality or results that do not exist yet.

Do not download a huge dataset merely to make the audit look complete. If real data is unavailable, distinguish code-level verification from end-to-end data verification.

## Recover the project history

Use commits and PRs to reconstruct the chronological development history.

For each existing PR, summarize:
- purpose;
- files added/changed;
- tests added;
- what capability became available after the PR;
- what the next logical unfinished step was.

Pay special attention to PRs #1–#4. Determine whether development stopped immediately after the dataset-loading layer.

## Output before any implementation

Produce an `ARCHAEOLOGY_REPORT.md` containing:

### 1. Executive summary
Explain in plain language what this project is, what problem it is intended to solve, and how far it actually got.

### 2. Current architecture
Show the real current repository structure and data flow.

### 3. Implemented vs planned matrix
Use columns:
`Component | Status | Evidence | Tests | Missing work`

### 4. Test and reproducibility audit
Report only results you actually observed.

### 5. Historical reconstruction
Summarize the development sequence from the git/PR history.

### 6. Gaps and technical debt
List concrete issues only. Do not invent problems for the sake of making the report longer.

### 7. Resume honesty check
Separate statements that are currently safe to claim on a resume from statements that would be premature or false.

### 8. Recommended continuation plan
Propose the next 5–8 small, ordered PRs that would turn this into a portfolio-quality NLP project.

Each proposed PR should include:
- objective;
- files to add/change;
- tests required;
- acceptance criteria;
- dependencies on previous PRs.

Prioritize a minimal complete pipeline before optional sophistication.

A sensible target is eventually:
raw arXiv metadata
→ reproducible preprocessing
→ TF-IDF + Logistic Regression / Linear SVM baselines
→ shared evaluation
→ DistilBERT fine-tuning
→ baseline-vs-transformer comparison
→ error analysis
→ reproducible README/results.

Interpretability, SciBERT/RoBERTa, demos, and extensive notebooks are stretch goals and should not displace the core pipeline.

## Critical honesty constraints

- Never invent accuracy, F1, training time, dataset size, or model results.
- Never describe code that is only planned as already implemented.
- Never claim a model was trained unless you actually run it or find verifiable saved results.
- Never rewrite working code merely to make it look more sophisticated.
- Preserve the project's existing coding style and path-safe design unless there is a concrete reason to change it.
- Do not commit raw arXiv data, model checkpoints, secrets, or large generated artifacts.

## After the audit

Do not immediately rewrite the whole repository.

First finish and save `ARCHAEOLOGY_REPORT.md`.

Then, if the report finds no blocking ambiguity, continue development in small, reviewable increments starting with the earliest missing core component. Keep tests green after every increment. Do not skip directly to a Transformer before a reproducible baseline and evaluation harness exist.

At the end of the task, report:
- exact files changed;
- exact tests run and their results;
- what is now genuinely complete;
- what remains incomplete;
- the next recommended PR.