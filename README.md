# Goodreads Review Popularity — Stanford Paper Reproduction

This repository contains notebooks to reproduce the main feature and model experiments in Bridget Daly's *Text-Based Prediction of Book Review Popularity* (Stanford, 2021).

## Notebooks
1. `02_paper_feature_engineering.ipynb` — reads `filtered_reviews.csv` from Google Drive and builds paper-aligned linguistic features.
2. `03_paper_model_training.ipynb` — compares feature subsets A–D with Logistic Regression, XGBoost and a neural network.
3. `04_evaluation_and_deliverables.ipynb` — creates result tables, plots, a two-page PDF draft, a PowerPoint draft, README and Streamlit demo.

## Data
Place `filtered_reviews.csv` in the root of Google Drive (`MyDrive/filtered_reviews.csv`) or update `DATA_PATH` in Notebook 02. Do not commit the dataset to a public repository unless you have permission.

## Run order
Run Notebook 01 for preprocessing Dataset, Notebook 02 to completion, then Notebook 03, then Notebook 04. All outputs are saved to `MyDrive/goodreads_reproduction_results/`.

## Reproducibility
- Seed: 229.
- Split: approximately 70% train, 15% validation, 15% test.
- The `popular` target is inherited from the supplied CSV; verify its construction uses the paper's cutoff before claiming exact reproduction.
- Notebook 03 defaults to `QUICK_RUN=True` and `MAX_ROWS=150000` for a resource-safe pilot. Set `QUICK_RUN=False` and `MAX_ROWS=None` for the final full-data run.
- TF-IDF is capped at 10,000 terms in this Colab implementation to control memory. This is a documented deviation from the repository's uncapped TF-IDF vectorizer.
- Check `failed_experiments.csv` and `training_manifest.json` after training.

## Current run snapshot
- Rows used: 100000
- Experiments completed: 18 / 18
- QUICK_RUN: True

Do not report pilot metrics as full-dataset results. Update this README after the final run.
