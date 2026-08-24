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
evaluation. Notebook 4 fine-tunes a pretrained DistilBERT classifier and
compares all three approaches.

## Experimental protocol

The results currently shown in Notebooks 2–4 are the initial benchmark runs
with training seed `42`. Before running the robustness extension, the following
protocol is fixed:

- The cleaned train, validation, and test partitions always use split seed
  `42`.
- TextCNN candidates are screened with seeds `13`, `42`, and `73`; the locked
  source-specific configurations are then retrained with seeds `13`, `42`,
  `73`, `101`, and `137`. These seeds affect model initialization and training
  order, not the data partitions.
- SMS → Enron is the primary transfer direction because it directly answers
  the research question. Enron → SMS is a secondary reverse-direction check;
  both in-domain evaluations provide context.
- Hyperparameters are selected separately for each training source using only
  its training and validation data. Candidate configurations are ranked by
  validation macro-F1 so both classes contribute to selection.
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
  textcnn_results.csv
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

[Initial Kaggle notebook: 03 - TextCNN](https://www.kaggle.com/code/kaloyanbozukov/notebook-3-textcnn-cross-domain-experiment)

[Executed Kaggle notebook: 04 - DistilBERT fine-tuning](https://www.kaggle.com/code/kaloyanbozukov/notebook4?scriptVersionId=341691004)

1. Import the notebooks in numerical order.
2. Enable Internet access for the notebook.
3. Enable a GPU accelerator for Notebooks 3 and 4.
4. Choose **Run All**.

Notebook 3 deliberately runs 24 validation-only tuning fits followed by 10
locked reporting fits. They run sequentially and each TensorFlow model is
released before the next fit to keep Kaggle memory use bounded.

The first cell clones this repository into `/kaggle/working`, so Kaggle uses the
same versioned code from `src/`. The public Kaggle links are the execution
records for the GPU notebooks; the repository copies remain the versioned
sources.

The numerical findings below are the initial seed-42 benchmarks. They are kept
as reference results until the repeated-run extension is complete.

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
paired SMS in-domain-to-Enron transfer gap. Seed 42 is used only for
representative learning curves, confusion matrices, and error examples.

The following values are the historical seed-42 reference and remain here only
until the upgraded Kaggle notebook has been executed:

- SMS → SMS: F1 0.930.
- Enron → Enron: F1 0.992.
- SMS → Enron: F1 0.563, an improvement of 0.051 over the ML baseline.
- Enron → SMS: F1 0.266, an improvement of 0.028 over the ML baseline.
- The SMS → Enron transfer gap decreases from 0.436 to 0.367, although the
  cross-domain ROC-AUC of 0.540 shows that generalization remains limited.

The upgraded notebook exports separate tuning, per-seed, and aggregate CSV
artifacts. It does not overwrite the legacy `textcnn_results.csv` before
Notebook 4 is migrated to the multi-seed schema.

## Notebook 4: DistilBERT fine-tuning

Notebook 4 fine-tunes `distilbert-base-uncased` separately on SMS and Enron.
It uses the same four evaluations, balanced class weights, and F1-based model
selection. Models are trained sequentially and released from GPU memory between
runs. The final section compares DistilBERT with both previous approaches.

- SMS → SMS: F1 0.974, the best in-domain SMS result.
- Enron → Enron: F1 0.991, comparable with the other in-domain results.
- SMS → Enron: F1 0.557, above the 0.512 baseline but slightly below TextCNN's
  0.563; ROC-AUC 0.480 confirms weak transfer.
- Enron → SMS: F1 0.268 with recall 0.927 and precision 0.157, showing that the
  model identifies most spam but produces many false positives.
- Fine-tuning a pretrained Transformer improves some scores, but it does not
  remove the domain shift between short SMS messages and longer emails.

In the initial seed-42 benchmarks, all three approaches perform well in-domain
and deteriorate sharply across domains. TextCNN gives the best SMS → Enron F1,
while DistilBERT gives the best SMS → SMS F1. This evidence suggests that
training on SMS alone does not produce a reliable email spam classifier; the
locked robustness protocol will test whether that conclusion persists across
training seeds and controlled experimental conditions.


## References

- Yoon Kim. [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/), EMNLP 2014.
- Victor Sanh et al. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108), 2019.
- Shai Ben-David et al. [A theory of learning from different domains](https://link.springer.com/article/10.1007/s10994-009-5152-4), Machine Learning 2010.
