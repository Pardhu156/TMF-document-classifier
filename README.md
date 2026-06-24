# TMF Classifier

Stage 1 is a small, modular BioClinicalBERT pipeline for classifying Trial Master File documents. It currently supports three classes:

- `protocol`
- `safety_report`
- `statistical_analysis_plan`

Long documents are split into text chunks because BERT has a maximum token limit. The data split happens at the document (`file_name`) level, so chunks from the same document never appear in both train and test sets. This prevents leakage.

## Workflow

1. Place raw PDF, DOCX, or TXT files in the three class folders under `data/`.
2. Run preprocessing to extract and clean text, create overlapping chunks, and save `artifacts/preprocessed_dataset.csv`, `artifacts/train.csv`, `artifacts/validation.csv`, and `artifacts/test.csv`.
3. Train on Colab/GPU when needed; the call in `main.py` is deliberately commented out.
4. Place the saved model and `label_encoder.pkl` inside `artifacts/` for local evaluation and prediction.

Training uses a document-level validation split for checkpoint selection; the test split is reserved for final evaluation. Evaluation reports chunk-level accuracy and macro F1, then groups chunk predictions by `file_name` and applies majority voting for document-level accuracy and macro F1. Metrics, run metadata, and both confusion matrices are stored in `artifacts/`.

## Local use

Install dependencies with `pip install -r requirements.txt`, then run:

```bash
python main.py
```

The local workflow is designed to use an already trained model in `artifacts/saved_bioclinicalbert_tmf_3class/`. Training is best run in Colab because fine-tuning BioClinicalBERT benefits from a GPU.

## Layout

```text
src/                 Modular preprocessing, training, evaluation, and prediction code
artifacts/           Preprocessed data, train/test splits, model, and evaluation outputs
data/<class>/         Raw files in protocol, safety_report, and statistical_analysis_plan folders
logs/                Application logs
```

## Stage 2: MLOps Foundation

Stage 2 adds optional MLflow experiment tracking through DagsHub, DVC data versioning guidance, structured metadata in `metadata/`, and `.env`-based secret management. Local execution remains safe when DagsHub credentials are not configured: the pipeline logs that MLflow is skipped and continues.

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env manually with your DagsHub values and access token.
python main.py
```

Never commit `.env` or a DagsHub token. `MLFLOW_TRACKING_PASSWORD` must contain a DagsHub access token, not an account password.

### Included in Stage 2

- MLflow and DagsHub experiment-tracking configuration
- DVC data-versioning setup instructions
- Structured dataset, model, training, evaluation, and version-history metadata
- `.env`-based secret management
- Lightweight configuration, metadata, and MLflow safety tests

## DVC + DagsHub Setup

Run these commands manually after creating a DagsHub repository:

```bash
dvc init

dvc add data/
dvc add artifacts/preprocessed_dataset.csv
dvc add artifacts/train.csv
dvc add artifacts/test.csv

git add data.dvc artifacts/preprocessed_dataset.csv.dvc artifacts/train.csv.dvc artifacts/test.csv.dvc .gitignore .dvc/config
git commit -m "Track datasets with DVC"

dvc remote add origin https://dagshub.com/<username>/<repo>.dvc

dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <dagshub_token>

dvc push
```

Git stores small `.dvc` metadata files; DagsHub stores the large data files. Never commit a DagsHub token. Track `artifacts/validation.csv` with DVC too because it is part of the model-selection split.
