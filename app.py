import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn
import unittest
import logging
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['MLFLOW_TRACKING_URI'] = 'file:./mlruns'

class IrisDataProcessor:
    def __init__(self):
        self.data = None
        self.target = None
        self.feature_names = None
        self.scaler = StandardScaler()
        logger.info("Initialized IrisDataProcessor")
    
    def prepare_data(self, test_size=0.2, random_state=42):
        iris = load_iris()
        self.feature_names = iris.feature_names
        # Dataframe Conversion
        self.data = pd.DataFrame(iris.data, columns=self.feature_names)
        self.target = pd.Series(iris.target, name='target')
        #Scale features
        scaled_features = self.scaler.fit_transform(self.data)
        self.data = pd.DataFrame(scaled_features, columns=self.feature_names)
        #Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    
    def get_feature_stats(self):
        stats = {
            'mean': self.data.mean().to_dict(),
            'std': self.data.std().to_dict(),
            'min': self.data.min().to_dict(),
            'max': self.data.max().to_dict()
        }
        logger.info("Feature statistics calculated")
        return stats

class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        self.best_model = None
        self.best_accuracy = 0
        
        #MLflow experiment
        mlflow.set_experiment("iris_classification")
        logger.info("Initialized IrisExperiment")
    
    def run_experiment(self):
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data()
        
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                model.fit(X_train, y_train)
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                # Predictions and metrics
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                # Log metrics
                self.log_results(model, {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'precision': precision,
                    'recall': recall
                })
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = model
                
                logger.info(f"Completed experiment for {model_name} with accuracy: {accuracy:.4f}")
    
    def log_results(self, model, metrics):
        """
        Log experiment results to MLflow
        """
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(model.get_params())

class IrisModelOptimizer:
    def __init__(self, experiment):
        self.experiment = experiment
        logger.info("Initialized IrisModelOptimizer")
    
    def quantize_model(self):
        if isinstance(self.experiment.best_model, LogisticRegression):
            self.experiment.best_model.coef_ = np.round(self.experiment.best_model.coef_, 3)
            self.experiment.best_model.intercept_ = np.round(self.experiment.best_model.intercept_, 3)
            logger.info("Model quantization completed")
        else:
            logger.warning("Quantization only supported for LogisticRegression")
    
    def run_tests(self):
        test_suite = unittest.TestLoader().loadTestsFromTestCase(IrisModelTests)
        unittest.TextTestRunner(verbosity=2).run(test_suite)

class IrisModelTests(unittest.TestCase):
    def setUp(self):
        self.processor = IrisDataProcessor()
        self.X_train, self.X_test, self.y_train, self.y_test = self.processor.prepare_data()
    
    def test_data_shape(self):
        self.assertEqual(len(self.processor.feature_names), 4)
        self.assertEqual(self.X_train.shape[1], 4)
    
    def test_scaling(self):
        scaled_mean = np.abs(self.X_train.mean()).mean()
        self.assertLess(scaled_mean, 0.05)
    
    def test_model_prediction(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

def main():
    try:
        processor = IrisDataProcessor()
        experiment = IrisExperiment(processor)
        experiment.run_experiment()
        optimizer = IrisModelOptimizer(experiment)
        optimizer.quantize_model()
        optimizer.run_tests()
        logger.info("ML pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}")
        raise
if __name__ == "__main__":
    main()