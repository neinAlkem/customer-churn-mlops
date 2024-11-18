import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
import joblib
import os
import yaml
import mlflow
import mlflow.sklearn

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog()

# Load dataset
data_train = pd.read_csv(config['data']['processed'])

# Splitting features and target
X = data_train.drop(columns=['Attrition_Flag'])
y = data_train['Attrition_Flag']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config['training']['test_size'], random_state=config['model']['random_state']
)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=config['training']['smote_random_state'])
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Start MLflow run
with mlflow.start_run():
    # Log configuration parameters
    mlflow.log_param("test_size", config['training']['test_size'])
    mlflow.log_param("smote_random_state", config['training']['smote_random_state'])
    mlflow.log_param("model_random_state", config['model']['random_state'])

    # Model training
    model = ExtraTreesClassifier(random_state=config['model']['random_state'])
    model.fit(X_train_smote, y_train_smote)

    # Save model locally
    os.makedirs(config['training']['model_save_path'], exist_ok=True)
    model_path = os.path.join(config['training']['model_save_path'], 'bank_churn_model.pkl')
    joblib.dump(model, model_path)

    # Log the trained model to MLflow
    mlflow.sklearn.log_model(model, "extra_trees_model")

    # Model predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

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
