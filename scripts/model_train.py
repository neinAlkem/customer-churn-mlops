import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report
import os
import json
import yaml

with open('config.yaml', 'r') as file :
    config = yaml.safe_load(file)

data_train = pd.read_csv(config['data']['processed'])

X = data_train.drop(columns=['Attrition_Flag'])  
y = data_train['Attrition_Flag'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['training']['test_size'], random_state=config['model']['random_state'])
smote = SMOTE(random_state=config['training']['smote_random_state'])
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = ExtraTreesClassifier(random_state=config['model']['random_state'])
model.fit(X_train_smote, y_train_smote)

os.makedirs(config['training']['model_save_path'], exist_ok=True)
joblib.dump(model, os.path.join(config['training']['model_save_path'], 'bank_churn_model.pkl'))

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

os.makedirs(config['evaluation']['save_path'], exist_ok=True)
evaluation_metrics = {
    'accuracy': accuracy,
    'classification_report': classification_rep
}

with open(os.path.join(config['evaluation']['save_path'], 'evaluation_metrics.json'), 'w') as f:
    json.dump(evaluation_metrics, f, indent=4)
print(f"Evaluation metrics saved to {config['evaluation']['save_path']}/evaluation_metrics.json")
