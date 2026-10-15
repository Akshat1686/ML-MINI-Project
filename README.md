# Text-Based Prediction of Book Review Popularity

A machine-learning mini-project that studies whether a Goodreads book review is likely to be **popular**, using review text together with review-level and reviewer/book-related features.

> **Project status:** This README documents the intended end-to-end workflow and the notebook stages developed for the project. Run the notebooks in order and use the generated result files for final metrics. Do not report scores or dataset counts until they have been produced by your own run.

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Dataset and Target Definition](#3-dataset-and-target-definition)
- [4. Methodology and Stages](#4-methodology-and-stages)
- [5. Models and Experiments](#5-models-and-experiments)
- [6. Project Structure](#6-project-structure)
- [7. Requirements](#7-requirements)
- [8. Setup](#8-setup)
- [9. Running the Project](#9-running-the-project)
- [10. Outputs and Evaluation](#10-outputs-and-evaluation)
- [11. Important Implementation Notes](#11-important-implementation-notes)
- [12. Limitations and Future Improvements](#12-limitations-and-future-improvements)
- [13. Reference](#13-reference)

## 1. Project Overview

Online book communities contain a large number of reviews, but only a small proportion attract substantial reader engagement. This project investigates whether information available in a review can help classify it as **popular** or **unpopular**.

The workflow combines:

- Review-text cleaning and NLP preprocessing.
- Text representations using Bag of Words (BoW) and TF-IDF.
- Numeric and linguistic features, such as review length and sentiment.
- Supervised classification models.
- Validation and test evaluation, model comparison, and feature-importance analysis.

The project is inspired by the Stanford CS229 report listed in the References section. The goal is to reproduce the paper's core experimental approach as closely as the available dataset and computing resources allow, while documenting any implementation differences.

## 2. Problem Statement

**Input:** A Goodreads review and the features available for that review.

**Output:** A binary prediction indicating whether the review is popular.

Popularity is not the same as a book rating. A rating measures how the reviewer evaluates the book; popularity measures the attention or engagement a review receives. Therefore, the star rating is treated as a possible predictor, not as the popularity label.

This is a **binary classification** task. The model learns from historical reviews and predicts a popularity class for reviews it has not seen during training.

## 3. Dataset and Target Definition

### Dataset

The project uses the Goodreads review dataset, including the compressed JSON Lines file:

`goodreads_reviews_dedup.json.gz`

The full dataset is large. It should be read incrementally in chunks/streaming batches rather than loaded as one enormous Pandas DataFrame. Keep the raw dataset outside GitHub unless repository size and dataset redistribution permissions explicitly allow it.

The review records may contain fields such as review text, book/user identifiers, rating, review date, `n_votes`, and `n_comments`. Actual fields can vary by dataset version, so inspect the input schema before running feature engineering.

### Popularity target

The target must be constructed from **engagement information**, not from the star rating.

The Stanford report defines review popularity using a review's share of the combined likes/comments received by reviews of the same book, then uses a threshold of `0.02` to create a binary label:

- `popular = 1` when the review's engagement share is greater than `0.02`.
- `popular = 0` otherwise.

For a reproduction, calculate the engagement count from the valid engagement fields available in the dataset (for example, `n_votes` and `n_comments`) and calculate the within-book share using the corresponding total for that book. Confirm the field meanings and the exact target construction in the feature-engineering code before training. If the available data does not support the paper's target exactly, document the alternative target explicitly rather than silently substituting rating or inventing engagement values.

Reviews without usable engagement information, invalid records, and records that cannot be assigned a valid target should be handled explicitly and their counts reported in the final results.

### Dataset audit to record

When running the pipeline, record the following in the report or results:

- Number of source records processed.
- Number of valid reviews retained.
- Missing/empty review-text count.
- Missing or invalid engagement/target count.
- Duplicate-handling policy and number removed, if deduplication is performed.
- Number of popular and unpopular reviews before and after any training-only resampling.
- Final training, validation, and test counts.

Do not place guessed counts in this README.

## 4. Methodology and Stages

### Stage 1 — Dataset inspection and exploratory analysis

1. Read the compressed JSON Lines data in batches.
2. Inspect available columns, data types, missing values, duplicate records, and engagement fields.
3. Examine the distribution of review length, ratings, comments/votes, and the popularity label.
4. Check class imbalance. Popular reviews may be much less common than unpopular reviews.
5. Save checkpoints and intermediate outputs to disk so the full raw dataset does not need to remain in memory.

### Stage 2 — Cleaning and preprocessing

1. Validate that each record has usable review text.
2. Normalize text (including lowercasing where appropriate).
3. Remove or normalize HTML markup and unwanted characters while preserving meaningful linguistic signals.
4. Tokenize the text.
5. Apply stopword handling and lemmatization where used by the notebook.
6. Preserve meaningful negation and sentiment terms such as *not*, *never*, *bad*, and *excellent*.
7. Handle missing numeric fields explicitly rather than assuming they are zero.

The preprocessing choices should be consistent between training and inference. Do not apply transformations at prediction time that differ from those used to train the model.

### Stage 3 — Feature engineering

The implementation explores feature groups corresponding to the paper's experiments:

- **Non-textual features:** reviewer activity and available rating-related metadata, including rating differences where the required fields exist.
- **Textual/linguistic features:** word count, average word length, sentence-related measures, part-of-speech proportions, punctuation/quotation-related features, and sentiment.
- **Text representations:** Bag of Words and TF-IDF representations of review text.

NLTK is used for relevant NLP processing, and VADER sentiment features are included in the feature-engineering workflow. Feature creation should be run in chunks and written incrementally to disk. The generated feature table is expected to be saved as:

`MyDrive/goodreads_reproduction_results/goodreads_paper_features.csv`

This path is in Google Drive when running in Google Colab. Change it to a suitable local/project path if running elsewhere.

### Stage 4 — Feature subsets

The experiments compare different combinations of feature groups. In the notebooks, the subsets are referred to as **A, B, C, and D**. Keep their definitions consistent across training and evaluation, and document them in the results table.

The Stanford report compares non-textual features, textual features, and text representations separately and in combinations. Refer to the feature-subset definitions in the implementation notebook rather than assuming that every subset contains the same columns.

### Stage 5 — Train/validation/test split

1. Separate the dataset into training, validation, and test sets.
2. Use the validation set for model selection and hyperparameter decisions.
3. Keep the test set untouched until final evaluation.
4. Fit vectorizers and other learned preprocessing steps using the training data only; use the fitted transformations to transform validation/test data.
5. If class undersampling is required, apply it **only to the training split**. Never undersample or otherwise rebalance the validation or test sets.

This prevents data leakage and keeps the final evaluation representative of the original data distribution.

### Stage 6 — Model training

The project compares three main classifier families:

1. Logistic Regression.
2. Gradient Boosting using XGBoost.
3. A feed-forward Neural Network.

The models are trained with the applicable feature subsets and text representations. Regularization, tree settings, and neural-network training duration are selected using validation results. The final test set is used only after model selection.

### Stage 7 — Evaluation and interpretation

Evaluate the final models on held-out test data. Because the target can be imbalanced, do not rely on accuracy alone. Record:

- Accuracy.
- Sensitivity/recall for the popular class.
- Specificity for the unpopular class.
- ROC-AUC.
- Confusion matrix, where available.

Compare models and feature subsets using the same split and evaluation definitions. Inspect feature coefficients or model-specific feature importance where supported. Treat these as associations learned by the model, not proof that a feature causes popularity.

### Stage 8 — Reproducibility and deliverables

1. Preserve the notebooks and scripts needed to reproduce preprocessing, training, and evaluation.
2. Save model metrics, plots, and relevant logs.
3. Record the final dataset counts and the exact target definition.
4. Prepare the mini-project write-up, presentation, and live demonstration required by the course.
5. Ensure the GitHub repository includes setup and run instructions without committing large raw datasets, credentials, or temporary files.

## 5. Models and Experiments

| Model | Purpose | Notes |
|---|---|---|
| Logistic Regression | Interpretable baseline for binary classification | Use regularization where appropriate; coefficients can help interpret feature direction. |
| XGBoost / Gradient Boosting | Non-linear classification baseline | Tune tree depth, number of trees, and learning rate using validation data. |
| Feed-forward Neural Network | Compare a neural approach with classical models | Use a validation-based stopping/checkpoint strategy to reduce overfitting. |

Text representations to compare:

- **BoW:** represents a review using token counts.
- **TF-IDF:** weights terms according to their importance across the corpus.

The report should include the experiment combinations actually completed. Do not claim that every model/feature-subset combination ran successfully unless it is present in the saved outputs.

## 6. Project Structure

The following is a recommended logical layout. Keep the names aligned with the files actually committed to the repository.

```text
book-review-popularity/
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_and_deliverables.ipynb
├── data/
│   ├── raw/                 # Local only; normally excluded from Git
│   └── processed/           # Generated data/checkpoints; normally excluded
├── models/                  # Saved model artifacts, if exported
├── results/
│   ├── metrics.csv          # Actual metrics from completed runs
│   └── plots/               # Generated evaluation figures
├── requirements.txt
├── .gitignore
└── README.md
```

**Important:** This tree describes the intended organization, not a guarantee that every listed file or directory already exists. Remove or rename entries to match the repository before committing the README.

## 7. Requirements

### Software

- Python 3.10+ (use the version supported by the notebook environment).
- Jupyter Notebook or Google Colab.
- Pandas and NumPy for data processing.
- scikit-learn for preprocessing, splits, metrics, and Logistic Regression.
- NLTK for tokenization, stopwords, lemmatization, and linguistic features.
- VADER sentiment analysis through NLTK.
- XGBoost for gradient boosting.
- A neural-network framework used by the training notebook (for example, TensorFlow/Keras or PyTorch). Install the framework actually imported by the notebook.
- Matplotlib/Seaborn if the EDA and evaluation notebooks use them.

### Install dependencies

From the project root, create and activate a virtual environment if working locally, then install the packages listed in `requirements.txt`:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If using Google Colab, install any missing packages in a notebook cell and download the dataset to the runtime/Drive as required by the notebook. NLTK resources may need to be downloaded once in the environment, such as tokenizer, stopword, lemmatizer, and VADER lexicon resources. Follow the exact resource names requested by the code.

## 8. Setup

1. Clone the repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd <YOUR_REPOSITORY_DIRECTORY>
   ```

2. Obtain the Goodreads dataset from its authorized source and place it in the location expected by the notebook. The raw compressed dataset should not be committed to GitHub by default.

3. Open the notebooks in Jupyter or Google Colab.

4. Set the dataset path and output/checkpoint directory in the configuration cells. If using Google Drive in Colab, mount Drive before writing to `MyDrive`.

5. Check available disk space before processing the full dataset. Feature engineering can produce a large output file, and text vectorization/model training can require substantial RAM and CPU time.

6. Confirm that the engagement fields needed to construct the popularity label exist in the chosen dataset version. Do not proceed with an assumed target if the fields are absent.

## 9. Running the Project

Run the notebooks in sequence, adapting the names to the actual files in the repository:

1. **`01_Data_Preprocessing.ipynb` — Inspect the data**
   - Load the whole Dataset first.
   - Inspect schema, missing values, target ingredients, class balance, and text statistics.
   - Confirm the target definition and write down the cleaning policy.

2. **`02_feature_engineering.ipynb` — Build features**
   - Process the large JSONL file in chunks.
   - Clean review text and calculate the chosen textual, linguistic, sentiment, and metadata features.
   - Create the paper-inspired feature subsets and text representations.
   - Write intermediate results incrementally and verify that each chunk was processed successfully.

3. **`03_model_training.ipynb` — Train models**
   - Load the processed features.
   - Create or load the train/validation/test split.
   - Fit preprocessing/vectorization on training data only.
   - Train the applicable Logistic Regression, XGBoost, and neural-network experiments.
   - Select hyperparameters using validation results and save model artifacts if the notebook supports it.

4. **`04_evaluation_and_deliverables.ipynb` — Evaluate and prepare outputs**
   - Evaluate selected models on the untouched test set.
   - Generate the metrics table, confusion matrices/plots, and feature-importance outputs where supported.
   - Export actual results for the report and presentation.

If the notebooks are executed in Colab, reconnecting to a runtime may erase local runtime files. Save checkpoints and final outputs to persistent storage such as Google Drive.

## 10. Outputs and Evaluation

The completed project should produce the following evidence, where supported by the implementation:

- Dataset audit and final row counts.
- Feature-engineered data/checkpoints.
- Model comparison table for the experiments that ran.
- Test metrics: accuracy, sensitivity, specificity, and ROC-AUC.
- Confusion matrix and useful plots.
- Feature-importance/coefficient analysis.
- Saved model artifacts if model export is implemented.
- A concise write-up, presentation, and reproducible instructions for the demonstration.


## 11. Important Implementation Notes

- **No rating-as-label shortcut:** a star rating is not the popularity target.
- **No data leakage:** fit text vectorizers and learned preprocessing on training data only.
- **Training-only resampling:** if undersampling is used, apply it only to training data.
- **Memory management:** stream/chunk the raw JSONL data, save checkpoints, and avoid building a dense full-dataset TF-IDF matrix.
- **Sparse text features:** keep BoW/TF-IDF sparse wherever possible.
- **Target validation:** calculate engagement shares from valid engagement fields and document records that cannot be labelled.
- **Honest reporting:** use only the metrics and row counts generated by the actual run.
- **Reproducible outputs:** document package versions, random seeds, split strategy, feature-subset definitions, and hyperparameters in the notebooks/results.
- **Repository hygiene:** exclude raw datasets, credentials, large temporary files, notebook checkpoints, and environment folders using `.gitignore`.

## 12. Limitations and Future Improvements

- The definition of popularity depends on available engagement fields and the selected threshold.
- Dataset availability and schema may differ from those in the original paper.
- Class imbalance can make accuracy misleading; sensitivity, specificity, and ROC-AUC should be reviewed together.
- Text features may improve prediction only modestly over metadata, so compare feature groups fairly.
- More systematic hyperparameter tuning, calibration, additional metadata, or transformer-based text representations can be explored if time and compute permit.

## 13. Reference

Daly, Bridget. **“Text-Based Prediction of Book Review Popularity.”** Stanford University, CS229 project report.

https://cs229.stanford.edu/proj2021spr/report2/81893104.pdf
