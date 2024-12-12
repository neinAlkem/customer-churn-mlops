import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
import joblib
import os
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report
import json
from prometheus_client import start_http_server, Summary, Counter, Gauge
import time

# Start Prometheus metrics server
start_http_server(8000)

# Define Prometheus metrics
MODEL_TRAINING_TIME = Summary('model_training_seconds', 'Time spent training the model')
MODEL_TRAINING_COUNTER = Counter('model_training_attempts', 'Number of model training attempts')
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the trained model')
MODEL_TRAINING_SIZE = Gauge('model_training_size', 'Size of the training data')
MODEL_TEST_SIZE = Gauge('model_test_size', 'Size of the test data')

# Load configuration
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print(f"Error loading configuration: {e}")
    exit(1)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://172.25.176.1:5000")
mlflow.autolog()

# Load dataset
try:
    data_train = pd.read_csv(config['data']['processed'])
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Splitting features and target
X = data_train.drop(columns=['Attrition_Flag'])
y = data_train['Attrition_Flag']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config['training']['test_size'], random_state=config['model']['random_state']
)

# Update Prometheus metrics for data size
MODEL_TRAINING_SIZE.set(len(X_train))
MODEL_TEST_SIZE.set(len(X_test))

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=config['training']['smote_random_state'])
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train and evaluate the model with Prometheus integration
@MODEL_TRAINING_TIME.time()  # Measure training time
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    MODEL_TRAINING_COUNTER.inc()  # Increment training attempt counter
    
    # Train the model
    model = ExtraTreesClassifier(random_state=config['model']['random_state'])
    model.fit(X_train, y_train)
    
    # Save model locally
    os.makedirs(config['training']['model_save_path'], exist_ok=True)
    model_path = os.path.join(config['training']['model_save_path'], 'bank_churn_model.pkl')
    joblib.dump(model, model_path)
    
    # Log the trained model to MLflow
    mlflow.sklearn.log_model(model, "extra_trees_model")
    
    # Model predictions
    y_pred = model.predict(X_test)
    
    # Calculate and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    # Update Prometheus metrics
    MODEL_ACCURACY.set(accuracy)
    
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_dict(classification_rep, "classification_report.json")
    
    # Save evaluation metrics to JSON
    os.makedirs(config['evaluation']['save_path'], exist_ok=True)
    evaluation_metrics = {
        'accuracy': accuracy,
        'classification_report': classification_rep
    }
    with open(os.path.join(config['evaluation']['save_path'], 'evaluation_metrics.json'), 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)
    
    print(f"Evaluation metrics saved to {config['evaluation']['save_path']}/evaluation_metrics.json")
    
    print(f"Accuracy: {accuracy}")
    print(f"y_test: {y_test}")
    print(f"y_pred: {y_pred}")
    
    return model

# Start MLflow run and train the model
with mlflow.start_run():
    trained_model = train_and_evaluate_model(X_train_smote, y_train_smote, X_test, y_test)

# Prevent script from exiting
print("Prometheus metrics server is running. Press Ctrl+C to stop.")
while True:
    time.sleep(1)
