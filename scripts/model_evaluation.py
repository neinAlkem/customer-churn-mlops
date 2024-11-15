import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
import yaml

with open('config.yaml', 'r') as file :
    config = yaml.safe_load(file)
    
data_train = pd.read_csv(config['data']['processed'])
X_test = data_train.drop(columns=['Attrition_Flag'])
y_test = data_train('Attrition_Flag')

model = joblib.load(os.path.join(config['training']['model_save_path'],'bank_churn_model.pkl'))
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred , output_dict=True)

evaluation_metrics = {
    'accuracy': accuracy,
    'classification_report': report
}
with open(os.path.join(config['evaluation']['save_path'], 'evaluation_metrics.json'), 'w') as f:
    json.dump(evaluation_metrics, f, indent=4)

