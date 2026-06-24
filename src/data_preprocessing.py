"""Extract, clean, chunk, balance, and split raw TMF documents."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.config import DataIngestionConfig, MLOpsConfig, MetadataConfig
from src.exception import CustomException
from src.logger import logger
from src.metadata_manager import create_dataset_metadata


SUPPORTED_CLASSES = ("protocol", "safety_report", "statistical_analysis_plan")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MINIMUM_TEXT_LENGTH = 100


def _extract_pdf_text(file_path: Path) -> str:
    import fitz  # PyMuPDF

    with fitz.open(file_path) as document:
        return "\n".join(page.get_text() for page in document)


def _extract_docx_text(file_path: Path) -> str:
    from docx import Document

    document = Document(file_path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _extract_text(file_path: Path) -> str:
    """Extract text from the file types inventoried in the research notebooks."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(file_path)
    if suffix in {".doc", ".docx"}:
        return _extract_docx_text(file_path)
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _clean_text(text: str) -> str:
    """Remove noise while retaining case, headings, and section numbering."""
    cleaned_lines = []
    for line in text.splitlines():
        normalized_line = re.sub(r"[ \t]+", " ", line).strip()
        if normalized_line:
            cleaned_lines.append(normalized_line)
    return "\n".join(cleaned_lines)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Create overlapping word chunks without altering the document's text content."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    step = chunk_size - chunk_overlap
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step) if words[start : start + chunk_size]]


def _read_raw_documents(raw_data_dir: Path) -> pd.DataFrame:
    """Read the three supported class folders into document-level rows."""
    documents: list[dict[str, str]] = []
    for class_name in SUPPORTED_CLASSES:
        class_directory = raw_data_dir / class_name
        if not class_directory.exists():
            logger.warning("Class folder is missing and will be skipped: %s", class_directory)
            continue

        candidate_files = sorted(path for path in class_directory.iterdir() if path.is_file() and not path.name.startswith("."))
        legacy_doc_files = [path for path in candidate_files if path.suffix.lower() == ".doc"]
        for file_path in legacy_doc_files:
            logger.warning("Skipping legacy .doc file %s; convert it to .docx before preprocessing.", file_path.name)
        files = [path for path in candidate_files if path.suffix.lower() in SUPPORTED_EXTENSIONS]
        logger.info("Found %d raw files for class '%s'.", len(files), class_name)
        for file_path in tqdm(files, desc=f"Extracting {class_name}", unit="file"):
            try:
                full_text = _clean_text(_extract_text(file_path))
            except Exception as error:
                logger.warning("Skipping unreadable file %s: %s", file_path.name, error)
                continue
            if len(full_text) < MINIMUM_TEXT_LENGTH:
                logger.warning("Skipping %s because extraction produced insufficient text.", file_path.name)
                continue
            documents.append(
                {
                    "file_name": file_path.relative_to(raw_data_dir).as_posix(),
                    "class": class_name,
                    "full_text": full_text,
                }
            )

    if not documents:
        raise ValueError(f"No readable documents found under {raw_data_dir}")
    found_classes = {document["class"] for document in documents}
    missing_classes = set(SUPPORTED_CLASSES).difference(found_classes)
    if missing_classes:
        raise ValueError(f"No readable documents found for required classes: {sorted(missing_classes)}")
    return pd.DataFrame(documents)


def _build_chunk_dataframe(documents: pd.DataFrame, config: DataIngestionConfig) -> pd.DataFrame:
    chunks: list[dict[str, str]] = []
    for document in documents.to_dict(orient="records"):
        for index, chunk in enumerate(_chunk_text(document["full_text"], config.chunk_size, config.chunk_overlap)):
            chunks.append(
                {
                    "file_name": document["file_name"],
                    "class": document["class"],
                    "chunk_id": f"{document['file_name']}__{index:04d}",
                    "chunk_text": chunk,
                }
            )
    if not chunks:
        raise ValueError("No chunks were created from the extracted documents.")
    return pd.DataFrame(chunks, columns=["file_name", "class", "chunk_id", "chunk_text"])


def _log_chunk_counts(dataframe: pd.DataFrame, message: str) -> None:
    counts = dataframe["class"].value_counts().sort_index().to_dict()
    logger.info("%s: %s", message, counts)


def _balance_chunks(dataframe: pd.DataFrame, random_state: int) -> pd.DataFrame:
    minimum_count = dataframe["class"].value_counts().min()
    balanced = [
        group.sample(n=minimum_count, random_state=random_state)
        for _, group in dataframe.groupby("class", sort=True)
    ]
    return pd.concat(balanced, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)


def _split_by_document(
    dataframe: pd.DataFrame,
    config: DataIngestionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    document_labels = dataframe.groupby("file_name", as_index=False)["class"].agg(
        lambda labels: labels.mode().iat[0]
    )
    class_counts = document_labels["class"].value_counts()
    if len(document_labels) < 2 or class_counts.min() < 3:
        raise ValueError("At least three documents per class are required for train/validation/test splitting.")
    held_out_size = config.validation_size + config.test_size
    if not 0 < held_out_size < 1:
        raise ValueError("validation_size + test_size must be greater than zero and less than one.")

    train_documents, held_out_documents, _, held_out_labels = train_test_split(
        document_labels["file_name"],
        document_labels["class"],
        test_size=held_out_size,
        random_state=config.random_state,
        stratify=document_labels["class"],
    )
    test_fraction_of_held_out = config.test_size / held_out_size
    validation_documents, test_documents = train_test_split(
        held_out_documents,
        test_size=test_fraction_of_held_out,
        random_state=config.random_state,
        stratify=held_out_labels,
    )
    train_df = dataframe[dataframe["file_name"].isin(train_documents)].reset_index(drop=True)
    validation_df = dataframe[dataframe["file_name"].isin(validation_documents)].reset_index(drop=True)
    test_df = dataframe[dataframe["file_name"].isin(test_documents)].reset_index(drop=True)
    overlap = (
        set(train_df["file_name"]).intersection(validation_df["file_name"])
        | set(train_df["file_name"]).intersection(test_df["file_name"])
        | set(validation_df["file_name"]).intersection(test_df["file_name"])
    )
    if overlap:
        raise RuntimeError(f"Document leakage detected: {sorted(overlap)[:5]}")
    return train_df, validation_df, test_df


def run_data_preprocessing(config: DataIngestionConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create artifact CSVs from raw class folders and return train/test chunks."""
    config = config or DataIngestionConfig()
    try:
        if not config.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {config.raw_data_dir}")

        documents = _read_raw_documents(config.raw_data_dir)
        chunk_df = _build_chunk_dataframe(documents, config)
        chunk_df.to_csv(config.preprocessed_data_path, index=False)
        _log_chunk_counts(chunk_df, "Chunks per class before balancing")
        logger.info("Saved %d chunks to %s", len(chunk_df), config.preprocessed_data_path)

        split_source = chunk_df
        if config.balance_classes:
            split_source = _balance_chunks(chunk_df, config.random_state)
            split_source.to_csv(config.balanced_data_path, index=False)
            logger.info("Saved balanced dataset to %s", config.balanced_data_path)
        _log_chunk_counts(split_source, "Chunks per class after balancing")

        train_df, validation_df, test_df = _split_by_document(split_source, config)
        train_df.to_csv(config.train_data_path, index=False)
        validation_df.to_csv(config.validation_data_path, index=False)
        test_df.to_csv(config.test_data_path, index=False)
        create_dataset_metadata(
            config,
            MetadataConfig(),
            MLOpsConfig(),
            chunk_df,
            train_df,
            test_df,
            validation_df,
        )
        logger.info(
            "Saved train/validation/test split: %d/%d/%d documents and %d/%d/%d chunks.",
            train_df["file_name"].nunique(),
            validation_df["file_name"].nunique(),
            test_df["file_name"].nunique(),
            len(train_df),
            len(validation_df),
            len(test_df),
        )
        return train_df, test_df
    except Exception as error:
        logger.exception("Data preprocessing failed")
        raise CustomException(error, sys.exc_info()) from error
