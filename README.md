# Cross-Domain Spam Detection

Can a spam classifier trained on SMS messages generalize to emails?

The project uses two public Hugging Face datasets:

- [jngb-labs/sms-spam](https://huggingface.co/datasets/jngb-labs/sms-spam)
- [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)

The experiment compares in-domain and cross-domain performance. Notebook 1
loads, cleans, validates, and explores the data. Notebook 2 establishes a
TF-IDF logistic-regression baseline. Notebook 3 introduces a TextCNN trained
from scratch on each domain. Notebook 4 fine-tunes a pretrained DistilBERT
classifier and compares all three approaches.

## Repository structure

```text
notebooks/
  01_data_loading_and_eda.ipynb
  02_tfidf_logistic_baseline.ipynb
  03_textcnn.ipynb
  04_distilbert_fine_tuning.ipynb
src/
  data.py
  distilbert.py
  modeling.py
  textcnn.py
tests/
  test_data.py
  test_distilbert.py
  test_modeling.py
  test_textcnn.py
```

## Run in VS Code on Windows

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-windows.txt
python -m pytest
```

Open the notebooks in numerical order and select the `.venv` kernel. Notebooks
1 and 2 can be run locally; the two neural-network notebooks are intended for
Kaggle GPU sessions.

`requirements-windows.txt` uses the Windows certificate store. It avoids SSL
errors on managed networks without disabling certificate verification.

## Run in Kaggle

[Executed Kaggle notebook: 01 - Data Loading and EDA](https://www.kaggle.com/code/kaloyanbozukov/01-data-loading-and-eda)

[Executed Kaggle notebook: 03 - TextCNN](https://www.kaggle.com/code/kaloyanbozukov/notebook-3-textcnn-cross-domain-experiment)

1. Import the notebooks in numerical order.
2. Enable Internet access for the notebook.
3. Enable a GPU accelerator for Notebooks 3 and 4.
4. Choose **Run All**.

The first cell clones this repository into `/kaggle/working`, so Kaggle uses the
same versioned code from `src/`. Download the executed notebook and replace the
copy in this repository before final submission.

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

The TextCNN learns task-specific word embeddings and local phrase patterns. It
uses the same splits and four train/test combinations as the baseline so the
comparison isolates the effect of the model architecture.

- SMS → SMS: F1 0.930.
- Enron → Enron: F1 0.992.
- SMS → Enron: F1 0.563, an improvement of 0.051 over the ML baseline.
- Enron → SMS: F1 0.266, an improvement of 0.028 over the ML baseline.
- The SMS → Enron transfer gap decreases from 0.436 to 0.367, although the
  cross-domain ROC-AUC of 0.540 shows that generalization remains limited.

## Notebook 4: DistilBERT fine-tuning

Notebook 4 fine-tunes `distilbert-base-uncased` separately on SMS and Enron.
It uses the same four evaluations, balanced class weights, and F1-based model
selection. Models are trained sequentially and released from GPU memory between
runs. The final section compares DistilBERT with both previous approaches.

## Data policy

Raw data is downloaded at runtime and is not committed to Git. All cleaning,
label mapping, and split decisions are implemented in `src/data.py` and
documented in the notebook.
