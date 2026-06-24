"""BioClinicalBERT fine-tuning utilities (intended for Colab/GPU use)."""

from __future__ import annotations

import inspect
import sys

import joblib
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import DataIngestionConfig, ModelTrainingConfig
from src.exception import CustomException
from src.logger import logger


def _compute_metrics(prediction: object) -> dict[str, float]:
    logits, labels = prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def run_training(
    data_config: DataIngestionConfig | None = None,
    training_config: ModelTrainingConfig | None = None,
):
    """Fine-tune BioClinicalBERT and return trainer, test data, and label encoder."""
    data_config = data_config or DataIngestionConfig()
    training_config = training_config or ModelTrainingConfig()
    try:
        if not data_config.train_data_path.exists() or not data_config.test_data_path.exists():
            raise FileNotFoundError("Run data preprocessing before training.")

        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        train_df = pd.read_csv(data_config.train_data_path).dropna(subset=["chunk_text", "class"])
        test_df = pd.read_csv(data_config.test_data_path).dropna(subset=["chunk_text", "class"])
        train_df["chunk_text"] = train_df["chunk_text"].astype(str)
        test_df["chunk_text"] = test_df["chunk_text"].astype(str)

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["class"].astype(str))
        unknown_test_labels = set(test_df["class"].astype(str)).difference(label_encoder.classes_)
        if unknown_test_labels:
            raise ValueError(f"Test set contains labels absent from train set: {sorted(unknown_test_labels)}")
        train_df["labels"] = label_encoder.transform(train_df["class"].astype(str))
        test_df["labels"] = label_encoder.transform(test_df["class"].astype(str))
        joblib.dump(label_encoder, training_config.label_encoder_path)

        logger.info("Loading tokenizer and model: %s", training_config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            training_config.model_name,
            num_labels=len(label_encoder.classes_),
            id2label={index: label for index, label in enumerate(label_encoder.classes_)},
            label2id={label: index for index, label in enumerate(label_encoder.classes_)},
        )

        def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
            return tokenizer(batch["chunk_text"], truncation=True, max_length=training_config.max_length)

        train_dataset = Dataset.from_pandas(train_df[["chunk_text", "labels"]], preserve_index=False).map(tokenize, batched=True)
        test_dataset = Dataset.from_pandas(test_df[["chunk_text", "labels"]], preserve_index=False).map(tokenize, batched=True)
        columns = ["input_ids", "attention_mask", "labels"]
        train_dataset.set_format(type="torch", columns=columns)
        test_dataset.set_format(type="torch", columns=columns)

        argument_values = dict(
            output_dir=str(training_config.output_dir),
            learning_rate=training_config.learning_rate,
            per_device_train_batch_size=training_config.train_batch_size,
            per_device_eval_batch_size=training_config.eval_batch_size,
            num_train_epochs=training_config.num_epochs,
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            report_to="none",
            disable_tqdm=False,
            load_best_model_at_end=False,
        )
        signature = inspect.signature(TrainingArguments.__init__).parameters
        argument_values["eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"] = "epoch"
        training_arguments = TrainingArguments(**argument_values)
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=_compute_metrics,
        )
        logger.info("Starting training for %d epoch(s).", training_config.num_epochs)
        trainer.train()
        trainer.save_model(str(training_config.save_model_dir))
        tokenizer.save_pretrained(str(training_config.save_model_dir))
        logger.info("Saved trained model to %s", training_config.save_model_dir)
        return trainer, test_dataset, test_df, label_encoder
    except Exception as error:
        logger.exception("Training failed")
        raise CustomException(error, sys.exc_info()) from error
