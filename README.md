# TMF Classifier

Stage 1 is a small, modular BioClinicalBERT pipeline for classifying Trial Master File documents. It currently supports three classes:

- `protocol`
- `safety_report`
- `statistical_analysis_plan`

Long documents are split into text chunks because BERT has a maximum token limit. The data split happens at the document (`file_name`) level, so chunks from the same document never appear in both train and test sets. This prevents leakage.

## Workflow

1. Place raw PDF, DOCX, or TXT files in the three class folders under `data/`.
2. Run preprocessing to extract and clean text, create overlapping chunks, and save `artifacts/preprocessed_dataset.csv`, `artifacts/train.csv`, and `artifacts/test.csv`.
3. Train on Colab/GPU when needed; the call in `main.py` is deliberately commented out.
4. Place the saved model and `label_encoder.pkl` inside `artifacts/` for local evaluation and prediction.

Evaluation reports chunk-level accuracy and macro F1, then groups chunk predictions by `file_name` and applies majority voting for document-level accuracy and macro F1. Metrics, run metadata, and both confusion matrices are stored in `artifacts/`.

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
