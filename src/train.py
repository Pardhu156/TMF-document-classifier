"""BioClinicalBERT fine-tuning utilities (intended for Colab/GPU use)."""

from __future__ import annotations

import inspect
import sys
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import DataIngestionConfig, MLOpsConfig, MetadataConfig, ModelTrainingConfig
from src.exception import CustomException
from src.logger import logger
from src.metadata_manager import create_model_metadata, create_training_run_metadata
from src.mlflow_tracking import end_mlflow_run, log_artifacts, log_metrics, log_params, start_mlflow_run
from src.utils import get_current_timestamp


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
) -> tuple[object, Dataset, pd.DataFrame, LabelEncoder]:
    """Fine-tune BioClinicalBERT and return trainer, test data, and label encoder."""
    data_config = data_config or DataIngestionConfig()
    training_config = training_config or ModelTrainingConfig()
    mlops_config = MLOpsConfig()
    metadata_config = MetadataConfig()
    run_id = str(uuid4())
    started_at = get_current_timestamp()
    mlflow_run = start_mlflow_run(run_name=f"train_{mlops_config.model_version}")
    try:
        required_paths = (
            data_config.train_data_path,
            data_config.validation_data_path,
            data_config.test_data_path,
        )
        if not all(path.exists() for path in required_paths):
            raise FileNotFoundError("Run data preprocessing to create train, validation, and test CSV files.")

        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        train_df = pd.read_csv(data_config.train_data_path).dropna(subset=["chunk_text", "class"])
        validation_df = pd.read_csv(data_config.validation_data_path).dropna(subset=["chunk_text", "class"])
        test_df = pd.read_csv(data_config.test_data_path).dropna(subset=["chunk_text", "class"])
        train_df["chunk_text"] = train_df["chunk_text"].astype(str)
        validation_df["chunk_text"] = validation_df["chunk_text"].astype(str)
        test_df["chunk_text"] = test_df["chunk_text"].astype(str)

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["class"].astype(str))
        unknown_labels = set(validation_df["class"].astype(str)).union(test_df["class"].astype(str)).difference(label_encoder.classes_)
        if unknown_labels:
            raise ValueError(f"Validation/test labels absent from train set: {sorted(unknown_labels)}")
        train_df["labels"] = label_encoder.transform(train_df["class"].astype(str))
        validation_df["labels"] = label_encoder.transform(validation_df["class"].astype(str))
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
        validation_dataset = Dataset.from_pandas(validation_df[["chunk_text", "labels"]], preserve_index=False).map(tokenize, batched=True)
        test_dataset = Dataset.from_pandas(test_df[["chunk_text", "labels"]], preserve_index=False).map(tokenize, batched=True)
        columns = ["input_ids", "attention_mask", "labels"]
        train_dataset.set_format(type="torch", columns=columns)
        validation_dataset.set_format(type="torch", columns=columns)
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
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
        )
        signature = inspect.signature(TrainingArguments.__init__).parameters
        argument_values["eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"] = "epoch"
        training_arguments = TrainingArguments(**argument_values)
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=_compute_metrics,
        )
        logger.info("Starting training for %d epoch(s).", training_config.num_epochs)
        log_params(
            {
                "model_name": training_config.model_name,
                "num_epochs": training_config.num_epochs,
                "learning_rate": training_config.learning_rate,
                "train_batch_size": training_config.train_batch_size,
                "eval_batch_size": training_config.eval_batch_size,
                "max_length": training_config.max_length,
                "model_version": mlops_config.model_version,
                "dataset_version": mlops_config.dataset_version,
            }
        )
        trainer.train()
        validation_metrics = trainer.evaluate()
        log_metrics(validation_metrics, prefix="validation")
        trainer.save_model(str(training_config.save_model_dir))
        tokenizer.save_pretrained(str(training_config.save_model_dir))
        class_names = [str(label) for label in label_encoder.classes_]
        create_model_metadata(training_config, metadata_config, mlops_config, class_names)
        create_training_run_metadata(
            metadata_config,
            mlops_config,
            training_config,
            run_id,
            started_at,
            get_current_timestamp(),
            "completed",
            str(model.device),
            mlflow_run.info.run_id if mlflow_run is not None else None,
        )
        log_artifacts(
            [
                str(path)
                for path in (
                    training_config.save_model_dir / "config.json",
                    training_config.label_encoder_path,
                    metadata_config.model_metadata_path,
                    metadata_config.training_run_metadata_path,
                )
            ]
        )
        logger.info("Saved trained model to %s", training_config.save_model_dir)
        return trainer, test_dataset, test_df, label_encoder
    except Exception as error:
        create_training_run_metadata(
            metadata_config,
            mlops_config,
            training_config,
            run_id,
            started_at,
            get_current_timestamp(),
            "failed",
            None,
            mlflow_run.info.run_id if mlflow_run is not None else None,
        )
        logger.exception("Training failed")
        raise CustomException(error, sys.exc_info()) from error
    finally:
        end_mlflow_run()
