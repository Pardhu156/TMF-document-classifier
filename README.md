# TMF Document Classifier

An enterprise-style NLP system for automatically classifying clinical trial documents into their corresponding Trial Master File (TMF) sections using BioClinicalBERT.

## Overview

Clinical trials generate thousands of documents throughout their lifecycle. Manual classification and filing of these documents into the Trial Master File (TMF) is time-consuming, error-prone, and resource intensive.

This project automates TMF document classification by fine-tuning BioClinicalBERT on real-world clinical trial documents.

The system classifies documents into predefined TMF categories such as:

* Protocol
* Statistical Analysis Plan (SAP)
* Safety Report

## Features

* Fine-tuned BioClinicalBERT for TMF document classification.
* Automated PDF/DOCX text extraction.
* Document chunking for long clinical documents.
* Multi-class clinical document classification.
* Enterprise-ready modular architecture.
* Model evaluation using Accuracy, Macro F1-score, and Confusion Matrix.
* Extensible architecture for future TMF classes.

## Project Architecture

```text
Clinical Documents
        ↓
Text Extraction
        ↓
Preprocessing
        ↓
Chunking
        ↓
BioClinicalBERT
        ↓
TMF Section Prediction
```

## Dataset

The dataset consists of publicly available clinical trial documents collected from multiple sources, including:

* Clinical study protocols
* Statistical analysis plans
* Safety reports
* Public clinical research templates and regulatory documents

Documents were converted into text and segmented into overlapping chunks before training.

## Model

* Base Model: BioClinicalBERT
* Framework: Hugging Face Transformers
* Task: Multi-class Document Classification

## Evaluation Metrics

* Accuracy
* Macro F1 Score
* Precision
* Recall
* Confusion Matrix

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* BioClinicalBERT
* Scikit-learn
* Pandas
* FastAPI
* Docker
* MLflow

## Repository Structure

```text
TMF-Document-Classifier/
│
├── notebooks/
├── src/
├── app/
├── data/
├── models/
├── experiments/
├── requirements.txt
├── README.md
└── LICENSE
```

## Future Enhancements

* Add additional TMF classes (e.g., Informed Consent Forms, Clinical Study Reports).
* Agentic TMF auto-filing workflow.
* Confidence-based manual review pipeline.
* MLflow experiment tracking.
* Dockerized deployment.
* Cloud deployment support.