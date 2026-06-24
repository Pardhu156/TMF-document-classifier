"""Stage 1 entry point for local evaluation and prediction."""

from src.data_preprocessing import run_data_preprocessing
from src.evaluate import run_evaluation
from src.exception import CustomException
from src.logger import logger
from src.predict import predict_text
from src.train import run_training


if __name__ == "__main__":
    logger.info("TMF Classifier pipeline started")

    # Extract, clean, chunk, and split the raw documents under data/.
    try:
        train_df, test_df = run_data_preprocessing()
    except CustomException as error:
        logger.warning("Preprocessing skipped: %s", error)

    # Uncomment this line only when training on Colab/GPU:
    # trainer, test_dataset, test_df, label_encoder = run_training()

    # If model and artifacts/test.csv are already present, evaluate the saved model.
    try:
        evaluation = run_evaluation()
        logger.info("Saved evaluation metrics: %s", evaluation["metrics"])
    except CustomException as error:
        logger.warning("Saved model evaluation skipped: %s", error)

    sample_text = "This document describes study objectives, inclusion criteria, and treatment procedures."
    try:
        prediction = predict_text(sample_text)
        print(prediction)
    except CustomException as error:
        logger.warning("Prediction skipped: %s", error)
