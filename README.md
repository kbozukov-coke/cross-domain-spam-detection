# Cross-Domain Spam Detection

Can a spam classifier trained on SMS messages generalize to emails?

The project uses two public Hugging Face datasets:

- [jngb-labs/sms-spam](https://huggingface.co/datasets/jngb-labs/sms-spam)
- [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)

The main experiment will compare in-domain and cross-domain performance. Day 1
loads, cleans, validates, and explores the data. Day 2 adds a transparent
TF-IDF logistic-regression baseline before any neural model is introduced.

## Repository structure

```text
notebooks/
  01_data_loading_and_eda.ipynb
  02_tfidf_logistic_baseline.ipynb
src/
  data.py
  modeling.py
tests/
  test_data.py
  test_modeling.py
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

Open the notebooks in numerical order, select the `.venv` kernel, and choose
**Run All**.

`requirements-windows.txt` uses the Windows certificate store. It avoids SSL
errors on managed networks without disabling certificate verification.

## Run in Kaggle

[Executed Kaggle notebook: 01 - Data Loading and EDA](https://www.kaggle.com/code/kaloyanbozukov/01-data-loading-and-eda)

1. Import the notebooks in numerical order, starting with
   `notebooks/01_data_loading_and_eda.ipynb` and then
   `notebooks/02_tfidf_logistic_baseline.ipynb`.
2. Enable Internet access for the notebook.
3. Choose **Run All**.

The first cell clones this repository into `/kaggle/working`, so Kaggle uses the
same versioned code from `src/`. Download the executed notebook and replace the
copy in this repository before final submission.

## Day 1 findings

- SMS: 5,159 unique messages; about 12.4% spam.
- Cleaned Enron training source: 28,528 unique messages; about 48.1% spam.
- Median length: 61 characters for SMS and about 710 for Enron email.
- No exact normalized message is shared between the two sources.

## Day 2 baseline findings

- SMS → SMS: F1 0.948.
- Enron → Enron: F1 0.987.
- SMS → Enron: F1 0.512, a 0.436 drop from the SMS in-domain result.
- Enron → SMS: F1 0.238.
- The word-based baseline performs well in-domain but does not generalize well
  between short SMS messages and longer emails.

## Data policy

Raw data is downloaded at runtime and is not committed to Git. All cleaning,
label mapping, and split decisions are implemented in `src/data.py` and
documented in the notebook.
