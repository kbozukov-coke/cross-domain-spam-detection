# Cross-Domain Spam Detection

Can a spam classifier trained on SMS messages generalize to emails?

The project uses two public Hugging Face datasets:

- [jngb-labs/sms-spam](https://huggingface.co/datasets/jngb-labs/sms-spam)
- [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)

The experiment compares in-domain and cross-domain performance. Notebook 1
loads, cleans, validates, and explores the data, including tokenizer-specific
length and truncation rates. Notebook 2 establishes a TF-IDF
logistic-regression baseline with expanded metrics and test-sample bootstrap
intervals. Notebook 3 performs validation-only TextCNN tuning and repeated-seed
evaluation. Notebook 4 applies the same robustness standard to DistilBERT,
adds controlled length and class-count experiments, and compares all three
approaches.

## Experimental protocol

Notebook 3 contains the completed five-seed TextCNN experiment. Notebook 4 is
prepared for its expanded Kaggle run; its new result artifacts are not reported
until that execution finishes. The following protocol is fixed:

- The cleaned train, validation, and test partitions always use split seed
  `42`.
- TextCNN candidates are screened with seeds `13`, `42`, and `73`; the locked
  source-specific configurations are then retrained with seeds `13`, `42`,
  `73`, `101`, and `137`. These seeds affect model initialization and training
  order, not the data partitions.
- DistilBERT tuning is restricted to the primary SMS-trained model: three
  predeclared candidates are ranked with seeds `13`, `42`, and `73`. The
  secondary Enron-trained diagnostic keeps its reference configuration. Both
  locked source models are then evaluated with all five reporting seeds.
- SMS → Enron is the primary transfer direction because it directly answers
  the research question. Enron → SMS is a secondary reverse-direction check;
  both in-domain evaluations provide context.
- Candidate configurations are ranked by source-validation macro-F1 so both
  classes contribute to selection. TextCNN is tuned separately for both
  sources; DistilBERT HPO focuses on the primary SMS source to keep the
  experiment computationally proportionate.
- After selection, the configuration is locked before evaluation on the fixed
  test benchmarks. Test labels do not influence hyperparameter, checkpoint, or
  threshold selection.
- The headline result is spam-class F1 (`spam = 1`) for SMS → Enron. Final
  reporting will include every training seed rather than only the best run.

The test results were inspected in the initial notebooks. They are therefore
described as fixed confirmation benchmarks rather than pristine unseen
holdouts. They remain excluded from all new design and selection decisions.
The shared constants are defined in `src/protocol.py` so the notebooks cannot
silently use different splits, seeds, directions, or metric roles.

## Shared evaluation and controls

The robustness extension uses two model-independent utility modules:

- `src/evaluation.py` computes the original metrics together with macro-F1,
  balanced accuracy, MCC, PR-AUC, class support, Brier score, log loss, and
  equal-frequency calibration error. It also provides per-seed aggregation,
  stratified bootstrap intervals, paired model comparisons, reliability-table
  data, and one stable per-example prediction schema.
- `src/controls.py` creates deterministic class-count-matched training subsets
  without changing the prepared splits or mutating their data frames.

Control sampling uses seed `2026`; bootstrap resampling uses seed `20260821`.
These are deliberately separate from the five neural training seeds. Model
checkpoints are not part of this layer, and no training is performed by either
module.

## Repository structure

```text
notebooks/
  01_data_loading_and_eda.ipynb
  02_tfidf_logistic_baseline.ipynb
  03_textcnn.ipynb
  04_distilbert_fine_tuning.ipynb
src/
  controls.py
  data.py
  distilbert.py
  evaluation.py
  modeling.py
  protocol.py
  textcnn.py
results/
  textcnn_tuning_results.csv
  textcnn_tuning_summary.csv
  textcnn_seed_results.csv
  textcnn_seed_summary.csv
tests/
  test_controls.py
  test_data.py
  test_distilbert.py
  test_evaluation.py
  test_modeling.py
  test_protocol.py
  test_result_artifacts.py
  test_textcnn.py
```

## Run in Kaggle

[Executed Kaggle notebook: 01 - Data Loading and EDA](https://www.kaggle.com/code/kaloyanbozukov/notebook-1-data-loading-and-exploratory-analysis)

[Executed Kaggle notebook: 02 - TF-IDF baseline](https://www.kaggle.com/code/kaloyanbozukov/tf-idf-logistic-regression-baseline?scriptVersionId=344574378)

[Executed Kaggle notebook: 03 - TextCNN robustness experiment](https://www.kaggle.com/code/kaloyanbozukov/notebook3?scriptVersionId=344585113)

[Initial Kaggle benchmark: 04 - DistilBERT fine-tuning](https://www.kaggle.com/code/kaloyanbozukov/notebook4?scriptVersionId=341691004)

1. Import the notebooks in numerical order.
2. Enable Internet access for the notebook.
3. Enable a GPU accelerator for Notebooks 3 and 4.
4. Choose **Run All**.

Notebook 3 deliberately runs 24 validation-only tuning fits followed by 10
locked reporting fits. They run sequentially and each TensorFlow model is
released before the next fit to keep Kaggle memory use bounded.

Notebook 4 runs 9 SMS validation-only tuning fits, 10 locked reporting fits,
9 additional SMS length variants, and 5 Enron count-matched controls. The 33
fits run sequentially with immediate GPU cleanup; the expected dual-T4 runtime
is approximately 1.5 to 2 hours.

The first cell clones this repository into `/kaggle/working`, so Kaggle uses the
same versioned code from `src/`. The public Kaggle links are the execution
records for the GPU notebooks; the repository copies remain the versioned
sources.

Notebook 2 is deterministic and Notebook 3 reports five training seeds.
Notebook 4 must be re-executed in Kaggle before its robustness results replace
the initial seed-42 benchmark.

## Notebook 1 findings

- SMS: 5,159 unique messages; about 12.4% spam.
- Cleaned Enron training source: 28,528 unique messages; about 48.1% spam.
- Median length: 61 characters for SMS and about 710 for Enron email.
- No exact normalized message is shared between the two sources.

## Notebook 2 baseline findings

- SMS → SMS: F1 0.948.
- Enron → Enron: F1 0.987.
- SMS → Enron: F1 0.512, a 0.436 drop from the SMS in-domain result.
- Enron → SMS: F1 0.238.
- The word-based baseline performs well in-domain but does not generalize well
  between short SMS messages and longer emails.

## Notebook 3: TextCNN

The TextCNN learns task-specific word embeddings and local phrase patterns.
Four small, predeclared configurations are compared separately for SMS and
Enron using mean validation macro-F1 across three seeds. Sequence length stays
fixed at 256 for the later controlled length experiment. After both
configurations are locked, each is retrained with all five reporting seeds.
The notebook reports every seed, mean ± sample standard deviation, and the
paired SMS in-domain-to-Enron transfer gap. Seed 42 is singled out only for
representative learning curves, confusion matrices, and error examples.

Both sources select the `kernel_3` configuration. The locked five-seed results
are:

- SMS → SMS: F1 `0.915 ± 0.006`.
- Enron → Enron: F1 `0.992 ± 0.002`.
- SMS → Enron: F1 `0.526 ± 0.065`, versus `0.512` for the deterministic
  baseline.
- Only three of five SMS → Enron runs exceed the baseline; the mean change is
  `+0.014 ± 0.065`, so the apparent improvement is not consistent across
  seeds.
- Enron → SMS: F1 `0.272 ± 0.004`, versus `0.238` for the baseline.
- The primary transfer gap remains `0.388 ± 0.063`.

The old seed-42 SMS → Enron result of `0.563` was therefore a favorable single
run and overstated the stability of TextCNN's advantage. The multi-seed result
supports the broader conclusion of weak cross-domain generalization.

The repository includes the executed tuning, per-seed, and aggregate CSV
artifacts exported by the upgraded notebook. Notebook 4 consumes the five-seed
artifacts rather than the obsolete single-seed result.

## Notebook 4: DistilBERT fine-tuning

Notebook 4 fine-tunes `distilbert-base-uncased` separately on SMS and Enron.
It performs validation-only SMS HPO, five-seed evaluation of all four domain
pairs, a `64/128/256/512` max-length sensitivity experiment, and an Enron
training subset matched to the exact SMS class counts. It also reports
bootstrap F1 intervals, Brier score, log loss, ECE, and a representative
reliability diagram.

The earlier seed-42 run is retained only as motivation; its values are not used
as the final result. The expanded notebook exports separate tuning, per-seed,
length, and count-control CSV artifacts. This section will be updated with
mean ± sample standard deviation after the new Kaggle execution.

All approaches in the completed experiments perform well in-domain and
deteriorate sharply across domains. The five-seed TextCNN experiment shows that
its small average SMS → Enron advantage over TF-IDF is not consistent across
seeds. Notebook 4 now tests whether that conclusion survives stronger
DistilBERT controls; a zero-shot instruction-tuned generative model remains the
next extension after its artifacts are collected.


## References

- Yoon Kim. [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/), EMNLP 2014.
- Victor Sanh et al. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108), 2019.
- Shai Ben-David et al. [A theory of learning from different domains](https://link.springer.com/article/10.1007/s10994-009-5152-4), Machine Learning 2010.
