# Stage 1: BioClinicalBERT TMF Classifier

## Tech Stack

- Python
- BioClinicalBERT: `emilyalsentzer/Bio_ClinicalBERT`
- Hugging Face Transformers
- PyTorch
- pandas
- scikit-learn
- PDF/DOCX/TXT text extraction

## Key Steps

1. Defined a 3-class TMF classification problem.
2. Created a raw document folder structure by class.
3. Extracted and cleaned document text.
4. Split long documents into overlapping chunks.
5. Split train/validation/test data at document level to avoid leakage.
6. Fine-tuned/imported a BioClinicalBERT classifier.
7. Evaluated chunk-level and document-level performance.
8. Saved metrics, confusion matrix, model, tokenizer, and label encoder artifacts.

## Classes

- `protocol`
- `safety_report`
- `statistical_analysis_plan`

## Implementation Details

Long TMF documents are chunked because BERT models have a maximum sequence length. The project uses overlapping word chunks with:

- chunk size: 512
- chunk overlap: 50

Chunk predictions are aggregated into a document-level prediction using majority voting. Document-level confidence combines:

- model confidence
- vote confidence
- margin confidence between the top two classes

## Metrics

| Metric | Value |
| --- | ---: |
| Total documents | 44 |
| Total chunks | 2,151 |
| Chunk-level accuracy | 61.50% |
| Chunk-level macro F1 | 68.03% |
| Document-level accuracy | 77.78% |
| Document-level macro F1 | 75.00% |

## Files And Artifacts

| Path | Purpose |
| --- | --- |
| `src/data_preprocessing.py` | Extraction, cleaning, chunking, and data split |
| `src/training.py` | Training flow |
| `src/evaluation.py` | Evaluation and metrics |
| `src/prediction.py` | Text and chunk prediction helpers |
| `artifacts/metrics.json` | Classifier metrics |
| `artifacts/run_metadata.json` | Run metadata |
| `artifacts/saved_bioclinicalbert_tmf_3class/` | Saved model/tokenizer |
| `artifacts/label_encoder.pkl` | Label encoder |

## Limitations

- Small dataset size.
- Protocol recall needs improvement.
- Fine-tuning is best done on GPU, not local CPU.
- Model is a POC classifier, not clinically validated.
