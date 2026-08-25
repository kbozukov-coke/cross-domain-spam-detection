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
evaluation. Notebook 4 defines a robust DistilBERT protocol with controlled
length and class-count experiments. Notebook 5 adds a zero-shot generative
FLAN-T5 comparator with a fixed prompt, restricted-label confidence,
calibration analysis, and a small free-generation audit.

## Experimental protocol

The model comparisons follow a predeclared protocol:

- The cleaned train, validation, and test partitions always use split seed
  `42`.
- TextCNN candidates are screened with seeds `13`, `42`, and `73`; the locked
  source-specific configurations are then retrained with seeds `13`, `42`,
  `73`, `101`, and `137`. These seeds affect model initialization and training
  order, not the data partitions.
- SMS → Enron is the primary transfer direction because it directly answers
  the research question. Enron → SMS is a secondary reverse-direction check;
  both in-domain evaluations provide context.
- Candidate configurations are ranked by source-validation macro-F1 so both
  classes contribute to selection. TextCNN is tuned separately for both
  sources.
- After selection, the configuration is locked before evaluation on the fixed
  test benchmarks. Test labels do not influence hyperparameter, checkpoint, or
  threshold selection.
- The zero-shot experiment uses one predeclared `google/flan-t5-base` checkpoint
  pinned to revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` and one fixed
  prompt. It performs no project-specific training or prompt search and never
  shows train or validation examples to the model.
- Zero-shot uncertainty is a stratified bootstrap interval over test examples,
  not training-seed variability. Its SMS–Enron difference is a domain
  performance gap rather than a transfer gap.
- The headline result is spam-class F1 (`spam = 1`) for SMS → Enron. Final
  reporting includes every TextCNN training seed rather than only the best run.

The test results were inspected in the initial notebooks. They are therefore
described as fixed confirmation benchmarks rather than pristine unseen
holdouts. They remain excluded from all new design and selection decisions.
The shared constants are defined in `src/protocol.py` so the notebooks cannot
silently use different splits, seeds, directions, or metric roles.

## Related work

The TextCNN experiment follows Kim's sentence-classification architecture but
tests a different question: stability under a large SMS-to-email domain shift.
DistilBERT supplies a compact transformer benchmark and Notebook 4 specifies
stronger robustness controls. FLAN-T5 tests whether instruction tuning alone
supplies useful zero-shot transfer for spam classification. The interpretation
follows domain-adaptation research: strong source performance need not imply
low target error when the source and target distributions differ substantially.

## Shared evaluation and controls

The robustness work uses three model-independent utility modules:

- `src/evaluation.py` computes the original metrics together with macro-F1,
  balanced accuracy, MCC, PR-AUC, class support, Brier score, log loss, and
  equal-frequency calibration error. It also provides per-seed aggregation,
  stratified bootstrap intervals, paired model comparisons, reliability-table
  data, and one stable per-example prediction schema.
- `src/controls.py` creates deterministic class-count-matched training subsets
  without changing the prepared splits or mutating their data frames.
- `src/generative.py` implements length-normalized teacher-forced `ham`/`spam`
  scores, score-derived confidence, strict generated-label parsing, and
  generation compliance summaries. Its PyTorch imports are lazy, so the
  non-generative notebooks do not acquire GPU state from this module.

Control sampling uses seed `2026`; bootstrap resampling uses seed `20260821`.
These are deliberately separate from the five neural training seeds. Model
checkpoints are not part of this layer, and none of these utility modules trains
a model by itself.

## Repository structure

```text
notebooks/
  01_data_loading_and_eda.ipynb
  02_tfidf_logistic_baseline.ipynb
  03_textcnn.ipynb
  04_distilbert_fine_tuning.ipynb
  05_zero_shot_generative.ipynb
src/
  controls.py
  data.py
  distilbert.py
  evaluation.py
  generative.py
  modeling.py
  protocol.py
  textcnn.py
results/
  generative_generation_audit.csv
  generative_zero_shot_calibration.csv
  generative_zero_shot_confidence_coverage.csv
  generative_zero_shot_predictions.csv
  generative_zero_shot_results.csv
  textcnn_tuning_results.csv
  textcnn_tuning_summary.csv
  textcnn_seed_results.csv
  textcnn_seed_summary.csv
tests/
  test_controls.py
  test_data.py
  test_distilbert.py
  test_evaluation.py
  test_generative.py
  test_modeling.py
  test_protocol.py
  test_result_artifacts.py
  test_textcnn.py
```

## Run in Kaggle

[Executed Kaggle notebook: 01 - Data Loading and EDA](https://www.kaggle.com/code/kaloyanbozukov/notebook-1-data-loading-and-exploratory-analysis)

[Executed Kaggle notebook: 02 - TF-IDF baseline](https://www.kaggle.com/code/kaloyanbozukov/tf-idf-logistic-regression-baseline?scriptVersionId=344574378)

[Executed Kaggle notebook: 03 - TextCNN robustness experiment](https://www.kaggle.com/code/kaloyanbozukov/notebook3?scriptVersionId=344585113)

[Kaggle notebook: 04 - DistilBERT fine-tuning](https://www.kaggle.com/code/kaloyanbozukov/notebook4?scriptVersionId=341691004)

[Executed Kaggle notebook: 05 - zero-shot FLAN-T5](https://www.kaggle.com/code/kaloyanbozukov/notebook5?scriptVersionId=344857355)

1. Import the required notebook from this repository.
2. Enable Internet access for the notebook.
3. Enable a GPU accelerator for Notebooks 3, 4, and 5.
4. Choose **Run All**.

The notebooks share the frozen data protocol but can be executed independently.

Notebook 3 deliberately runs 24 validation-only tuning fits followed by 10
locked reporting fits. They run sequentially and each TensorFlow model is
released before the next fit to keep Kaggle memory use bounded.

Notebook 4 defines a 33-fit DistilBERT robustness protocol: 9 SMS
validation-only tuning fits, 10 locked reporting fits, 9 additional SMS length
variants, and 5 Enron count-matched controls. Its public Kaggle link records the
initial seed-42 benchmark.

Notebook 5 is independent of Notebook 4 artifacts. It runs one
`google/flan-t5-base` checkpoint on a single T4 in FP16, without training, and
uses batches of eight. The executed scoring pass took 7.2 seconds for SMS and
50.7 seconds for Enron, excluding model download and setup. Its optional
cross-notebook table is artifact-driven and does not affect the core exports.

The first cell clones this repository into `/kaggle/working`, so Kaggle uses the
same versioned code from `src/`. The public Kaggle links are the execution
records for the GPU notebooks; the repository copies remain the versioned
sources.

Notebook 2 is deterministic and Notebook 3 reports five training seeds.
Notebook 5 has been executed and its five result artifacts are stored in
`results/`.

Repository notebook outputs are intentionally cleared; the versioned Kaggle
runs above are the execution records and the compact CSV files in `results/`
are the machine-readable evidence.

## Local verification

The data is downloaded at runtime, so no raw dataset copy is committed. A local
CPU environment is sufficient for the unit and artifact tests:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

The neural experiments are intended for Kaggle GPU execution rather than local
training.

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
Enron using mean validation macro-F1 across three seeds. Sequence length is
fixed at 256 for comparability; Notebook 4 defines the separate controlled
length protocol. After both configurations are locked, each is retrained with
all five reporting seeds.
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
artifacts exported by the upgraded notebook. They replace the obsolete
single-seed result in the final comparison.

## Notebook 4: DistilBERT robustness protocol

Notebook 4 fine-tunes `distilbert-base-uncased` separately on SMS and Enron.
The protocol includes validation-only SMS HPO,
five-seed evaluation of all four domain pairs, a `64/128/256/512` max-length
sensitivity experiment, and an Enron training subset matched to the exact SMS
class counts. Its reporting code covers bootstrap F1 intervals, Brier score,
log loss, ECE, and a representative reliability diagram.

The completed supervised approaches perform well in-domain and deteriorate
sharply across domains. The five-seed TextCNN experiment shows that its small
average SMS → Enron advantage over TF-IDF is not consistent across seeds.
Notebook 4 provides the corresponding transformer robustness and control
protocol.

## Notebook 5: zero-shot FLAN-T5

Notebook 5 evaluates `google/flan-t5-base` as a domain-agnostic zero-shot
comparator on the complete SMS and Enron test sets. A single prompt defines
`ham` and `spam`; there are no demonstrations, prompt search, or parameter
updates. Because T5 tokenizes `ham` and `spam` into different numbers of
tokens, the primary prediction compares their mean decoder-token log
probability, including EOS, and then normalizes only over those two choices.
The resulting score-derived pseudo-probability is not a globally normalized
sequence probability and remains sensitive to the selected verbalizers.

The notebook reports the shared classification and calibration metrics, a 95%
stratified bootstrap interval for F1, confidence–coverage tables, measured
truncation, and a balanced free-generation audit of 50 examples per domain.
Invalid generated answers remain in the denominator. Five compact CSV
artifacts are exported without original message text, including calibration
and confidence–coverage tables. Core artifacts are saved before the optional
cross-notebook comparison, so Notebook 4 cannot prevent their export.

The executed zero-shot results show a strong bias toward `ham`:

- SMS: spam F1 `0.019`, 95% bootstrap CI `[0.000, 0.059]`, recall
  `0.010`, and ROC-AUC `0.680`. Only 8 of 774 messages are predicted as spam.
- Enron: spam F1 `0.062`, CI `[0.041, 0.085]`, recall `0.032`, and
  ROC-AUC `0.661`. Only 33 of 1,981 emails are predicted as spam.
- SMS accuracy is `0.868`, but balanced accuracy is `0.500`; the high raw
  accuracy mostly reflects the majority ham class rather than useful spam
  detection.
- No SMS message is truncated. Enron truncation affects 581 of 1,981 examples
  (`29.3%`), so long-email context remains a limitation rather than a complete
  explanation for the failure.
- In the balanced generation audit, valid one-label output is produced for
  `98%` of SMS and `68%` of Enron examples. Accuracy with invalid outputs
  counted as wrong is `48%` and `20%`, respectively. Agreement between valid
  generation and restricted-label predictions reflects their shared ham bias,
  not strong reliability.

For descriptive context, the zero-shot F1 scores are far below the SMS-trained
TF-IDF results (`0.948` on SMS and `0.512` on Enron) and the five-seed TextCNN
means (`0.915` and `0.526`). These protocols are not identical: FLAN-T5 has no
project training domain and its bootstrap interval measures test-sample
uncertainty rather than training-seed variation. The negative result is still
informative: this fixed-prompt, off-the-shelf generative model does not solve
spam detection, and its moderate ranking signal does not translate into useful
classification at the predeclared threshold.

## Final conclusion

The answer to the research question is **no, not reliably**. All trained models
perform strongly in-domain, but their spam F1 drops sharply after crossing the
SMS/email boundary. TextCNN's small mean advantage over TF-IDF is unstable
across seeds, and the fixed-prompt FLAN-T5 comparator is strongly biased toward
`ham`. The combined evidence points to domain mismatch—not model complexity
alone—as the central limitation. Deployment claims would require target-domain
labels, external validation, and additional adaptation experiments.


## References

- Yoon Kim. [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/), EMNLP 2014.
- Victor Sanh et al. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108), 2019.
- Hyung Won Chung et al. [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416), 2022.
- Google. [FLAN-T5 Base model card](https://huggingface.co/google/flan-t5-base).
- Shai Ben-David et al. [A theory of learning from different domains](https://link.springer.com/article/10.1007/s10994-009-5152-4), Machine Learning 2010.
