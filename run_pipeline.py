import logging
from stages.ingest_data import IngestData
from stages.clean_data import DataCleaner
from stages.split_data import DataSplitter
from stages.feature_engineering import FeatureEngineer
from stages.train_model import ModelTrainer
from stages.evaluate_model import Evaluator
from sklearn.linear_model import LogisticRegression

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Ingest Data
        data_path = 'Train_data.csv'
        ingestor = IngestData(data_path)
        df = ingestor.get_data()
        logging.info("Data ingested successfully.")
        logging.debug(f"Data sample: {df.head()}")

        # Clean Data
        cleaner = DataCleaner(scale_method='standardize')
        X = cleaner.clean_data(df)
        logging.info("Data cleaned successfully.")
        logging.debug(f"Cleaned data sample: {X.head()}")

        # Feature Selection & Deliver Data
        feature_engineer = FeatureEngineer(X)
        selected_features, y_target = feature_engineer.select_features(X, 'class', n_features_to_select=10)
        X_selected, target = feature_engineer.deliver_data(X, selected_features, y_target)
        logging.info("Features selected and data delivered successfully.")
        logging.debug(f"Selected features: {selected_features}")

        # Split Data
        datasplitter = DataSplitter()
        X_train, X_test, y_train, y_test = datasplitter.split_data(X_selected, target, 0.2)
        logging.info("Data split successfully.")
        logging.debug(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        # Train Model
        model = LogisticRegression()
        trainer = ModelTrainer(X_train, y_train, model)
        trainer.train_model()
        trained_model = trainer.get_trained_model()
        logging.info("Model trained successfully.")

        # Evaluate Model
        evaluator = Evaluator(X_test, y_test)
        evaluation_results = evaluator.evaluate(trained_model)
        logging.info("Model evaluated successfully.")
        logging.debug(f"Evaluation results: {evaluation_results}")

        # Save The Model
        trainer.save_trained_model("./saved_model/model.pkl")
        logging.info("Model saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
