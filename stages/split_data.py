import pandas as pd
import logging
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self) -> None:
        pass

    def split_data(self, dataset: pd.DataFrame, target: pd.Series, percentage: float) -> tuple:
        """
        Splits the dataset and target into training and test sets using sklearn's train_test_split.
        """
        try:
            if not (0 < percentage < 1):
                raise ValueError("Percentage must be between 0 and 1.")
            
            X = dataset
            y = target
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentage, random_state=42)
            
            logging.info(f"Data split into train and test sets with test size = {percentage}.")
            return X_train, X_test, y_train, y_test
        except ValueError as ve:
            logging.error(f"Value error: {ve}")
            raise ve
        except Exception as e:
            logging.error("Failed to split data.")
            raise e

    def split_data_v2(self, data: pd.DataFrame) -> tuple:
        """
        Custom logic to handle imbalanced datasets with fraud detection use case.
        Splits the data into training and test sets with the target variable.
        """
        try:
            # Splitting normal and fraud cases
            normal = data[data.target == 0].sample(frac=1).reset_index(drop=True)
            fraud = data[data.target == 1]

            # Defining training and test sets
            X_Train = normal.iloc[:200000].drop('target', axis=1)
            X_Test = pd.concat([normal.iloc[200000:], fraud]).sample(frac=1).reset_index(drop=True)

            logging.info(f"Data split into custom train and test sets with {len(X_Train)} training samples and {len(X_Test)} test samples.")
            return X_Train, X_Test
        except KeyError as ke:
            logging.error(f"Key error: {ke}. Ensure the 'target' column is in the dataset.")
            raise ke
        except Exception as e:
            logging.error("Failed to split data using custom logic.")
            raise e
